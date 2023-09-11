import copy
import math
import os
import random
import sys

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from scipy import ndimage

sys.path.append('..')
from util.box_ops import box_xyxy_to_cxcywh

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


class RectumDataloader(Dataset):
    def __init__(self, root_dir, mode, imgsize: tuple[int, int]):
        assert mode in ['train', 'test'], 'dataset should be either train set or test set'
        csv_file_dir = os.path.join(root_dir, mode, mode + '_bbox.csv')
        self.csv = pd.read_csv(csv_file_dir)
        self.root_dir = root_dir
        self.npz_dir = os.path.join(self.root_dir, mode, mode + '_npz')
        self.imgsize = imgsize
        self.train = True if mode == 'train' else False
        self.num_classes = 2

        # list_dir = os.path.join(root_dir, mode, mode + '_bbox.txt')
        # with open(list_dir, 'w') as f:
        #     for idx in range(self.__len__()):
        #         filename = self.csv.iloc[idx, 0]
        #         f.write(filename + '\n')

    def __len__(self):
        """return the length of the dataset"""
        return self.csv.shape[0] * 1  # duplicated with num_classes

    def __getitem__(self, idx):
        # read npz
        filename = self.csv.iloc[idx, 0]
        filename = os.path.join(self.npz_dir, filename + '.npz')

        npz = np.load(filename)
        img = npz['image']
        mask = npz['label']

        # assert img.min() >= 0 and img.max() <= 1

        # read bbox
        bbox = eval(self.csv.iloc[idx, 1])
        bbox = np.array(bbox, dtype=float)

        # 3-cls classification
        mask[mask > 2] = 2

        # data augmentation
        from albumentations import CLAHE
        img = np.uint8(img * 255)
        img = CLAHE(p=1)(image=img)['image']
        img = np.array(img, dtype=float) / 255
        if self.train:
            if random.random() > 0.5:
                img, mask, bbox = random_rot_flip(img, mask, bbox)
            elif random.random() > 0.5:
                img, mask, bbox = random_rotate(img, mask, bbox)

        if not np.sum(mask > 0) > 0:  # current frame doesn't contain mask of cls
            next_idx = (idx + 1) % (self.csv.shape[0] * self.num_classes)
            return self.__getitem__(next_idx)  # fetch the next frame

        # calculate new bbox accordding to new mask
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        h, w = mask.shape
        if self.train:  # add random margin
            x_min = max(0, x_min - np.random.randint(0, 5))
            x_max = min(w, x_max + np.random.randint(0, 5))
            y_min = max(0, y_min - np.random.randint(0, 5))
            y_max = min(h, y_max + np.random.randint(0, 5))
        else:  # add constant margin
            margin = 2
            x_min = max(0, x_min - margin)
            x_max = min(w, x_max + margin)
            y_min = max(0, y_min - margin)
            y_max = min(h, y_max + margin)
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=float)

        # reshape img & mask
        h, w = img.shape
        if w != self.imgsize[0] or h != self.imgsize[1]:
            img = zoom(img, (self.imgsize[0] / w, self.imgsize[1] / h), order=3)
            mask = zoom(mask, (self.imgsize[0] / w, self.imgsize[1] / h), order=0)

        if np.sum(mask > 0) == 0:
            return self.__getitem_multicls__(idx)

        points = None
        pts_per_cls = 10
        for cls_idx in range(1, self.num_classes + 1):
            # randomly pick 5 points in the mask
            posx, posy = np.where(mask == cls_idx)
            if len(posx) == 0:  # skip cls not included
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
            else:
                points = torch.cat([points, _points], dim=0)
        points = points.unsqueeze(0)  # [1, cls, 10, 2]
        if points.size(1) > 1:
            points = points[:, :, [0, 2, 4, 6, 8]]
        else:
            points = points.reshape(1, 2, 5, 2)

        # reshape bbox
        bbox[[0, 2]] *= self.imgsize[0] / w
        bbox[[1, 3]] *= self.imgsize[1] / h

        # convert to long
        labels = torch.tensor([1]).long()
        # box normalization
        h, w = img.shape
        bbox = torch.tensor(bbox).float().unsqueeze(0)
        orig_boxes = bbox.clone().detach()
        bbox[:, [0, 2]] /= w
        bbox[:, [1, 3]] /= h
        bbox = box_xyxy_to_cxcywh(bbox)

        orig_size = torch.tensor([w, h], dtype=torch.long)

        img = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1).float()
        mask = torch.tensor(mask).long()

        target = {'boxes': bbox,
                  'labels': labels,
                  'mask': mask,
                  'orig_size': orig_size,
                  'id': torch.tensor(idx),
                  'orig_boxes': orig_boxes,
                  'points': points}

        return img, target


if __name__ == '__main__':
    img_size = 224
    root = ''
    dataset_train = RectumDataloader(root, mode='train', imgsize=(img_size, img_size))
    dataset_val = RectumDataloader(root, mode='test', imgsize=(img_size, img_size))

    for img, _ in dataset_train:
        pass
