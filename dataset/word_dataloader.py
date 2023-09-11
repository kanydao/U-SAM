import os
import random
from os.path import join
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import math
from util.box_ops import box_xyxy_to_cxcywh


# 随机翻转
def random_rot_flip(image, label, box):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    x1, y1, x2, y2 = box
    w, h = image.shape
    if axis == 0:
        box = np.array([w - x2, y1, w - x1, y2])
    else:
        box = np.array([x1, h - y2, x2, h - y1])

    return image, label, box


# 正负20°随机翻转
def random_rotate(image, label, box):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    x1, y1, x2, y2 = box
    cos = math.cos(angle * math.pi / 180)
    sin = math.sin(angle * math.pi / 180)
    xx1 = cos * x1 - sin * y1
    yy1 = sin * x1 + cos * y1
    xx2 = cos * x2 - sin * y2
    yy2 = sin * x2 + cos * y2
    x1 = min(xx1, xx2)
    x2 = max(xx1, xx2)
    y1 = min(yy1, yy2)
    y2 = max(yy1, yy2)
    box = np.array([x1, y1, x2, y2])
    return image, label, box


class WordDataset(Dataset):
    def __init__(self, root_dir, mode, imgsize: tuple[int, int]):
        assert mode in ['train', 'test'], 'dataset should be either train set or test set'
        self.data_dir = os.path.join(root_dir, mode)
        self.gt_path = join(self.data_dir, 'npy_gts')
        self.img_path = join(self.data_dir, 'npy_imgs')
        self.npy_files = sorted(os.listdir(self.gt_path))
        self.imgsize = imgsize
        self.train = True if mode == 'train' else False
        self.num_classes = 16

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        slice_name = self.npy_files[idx]
        img_data_path = join(self.img_path, slice_name)
        gt_data_path = join(self.gt_path, slice_name)
        image, label = np.load(img_data_path), np.load(gt_data_path)
        image = image.astype(float) / 255
        roi_margin = 60
        image = image[roi_margin:-roi_margin, roi_margin:-roi_margin, 0]
        mask = label[roi_margin:-roi_margin, roi_margin:-roi_margin]

        bbox = np.array([0, 0, 0, 0], dtype=float)

        # data augmentation
        from albumentations import CLAHE
        image = np.uint8(image * 255)
        image = CLAHE(p=1)(image=image)['image']
        image = np.array(image, dtype=float)/255

        if self.train:
            if random.random() > 0.5:
                image, mask, bbox = random_rot_flip(image, mask, bbox)
            elif random.random() > 0.5:
                image, mask, bbox = random_rotate(image, mask, bbox)

        # calculate new bbox accordding to new mask
        assert np.sum(mask > 0) > 0
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        h, w = mask.shape
        x_min = max(0, x_min - np.random.randint(0, 20))    # add random margin
        x_max = min(w, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(h, y_max + np.random.randint(0, 20))
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=float)

        # reshape img & mask
        h, w = image.shape
        if w != self.imgsize[0] or h != self.imgsize[1]:
            image = zoom(image, (self.imgsize[0] / w, self.imgsize[1] / h), order=3)
            mask = zoom(mask, (self.imgsize[0] / w, self.imgsize[1] / h), order=0)

        # generate points as prompt
        points = None
        pts_labels = None
        pts_per_cls = 5
        for cls_idx in range(1, self.num_classes + 1):
            # randomly pick 5 points in the mask
            posx, posy = np.where(mask == cls_idx)
            if len(posx) < pts_per_cls:  # skip cls not included
                continue
            step = len(posx) // pts_per_cls
            if self.train:
                sample_idx = random.sample(range(len(posx)), pts_per_cls)
                _points = [[posx[i], posy[i]] for i in sample_idx]
            else:
                _points = [[posx[i * step], posy[i * step]] for i in range(pts_per_cls)]
            _points = torch.tensor(np.array(_points, dtype=int)).unsqueeze(0)
            if points is None:
                points = _points
                pts_labels = torch.ones(pts_per_cls, dtype=torch.int).unsqueeze(0)
            else:
                points = torch.cat([points, _points], dim=0)
                _pts_labels = torch.ones(pts_per_cls, dtype=torch.int).unsqueeze(0)
                pts_labels = torch.cat([pts_labels, _pts_labels], dim=0)
        points = points.unsqueeze(0)  # [1, cls, 10, 2]

        tile = math.ceil(self.num_classes / points.size(1))
        tmp = points
        for _ in range(tile):
            points = torch.cat([points, tmp], dim=1)
        points = points[:, :self.num_classes]

        # convert to long
        labels = torch.tensor([1]).long()
        # box normalization
        h, w = image.shape
        bbox = torch.tensor(bbox).float().unsqueeze(0)
        orig_boxes = bbox.clone().detach()
        bbox[:, [0, 2]] /= w
        bbox[:, [1, 3]] /= h
        bbox = box_xyxy_to_cxcywh(bbox)

        orig_size = torch.tensor([w, h], dtype=torch.long)

        image = torch.tensor(image).unsqueeze(0).repeat(3, 1, 1).float()
        mask = torch.tensor(mask).long()

        target = {'boxes': bbox,
                  'labels': labels,
                  'mask': mask,
                  'orig_size': orig_size,
                  'id': torch.tensor(idx),
                  'orig_boxes': orig_boxes,
                  'points': points}

        return image, target
