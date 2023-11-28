# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import json
import os.path
import random
import time
import datetime
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DistributedSampler, DataLoader
from backbone import UNet as downsample
import argparse
import util.misc as utils
import matplotlib.pyplot as plt


class mIoUAB(nn.Module):
    def __init__(self, n_classes):
        super(mIoUAB, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _iou_index(self, score, target):
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        index_a = intersect
        index_b = y_sum + z_sum - intersect
        return index_a, index_b

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        inputs = self._one_hot_encoder(inputs)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), \
            'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        index_a = []
        index_b = []
        for i in range(self.n_classes):
            a, b = self._iou_index(inputs[:, i], target[:, i])

            index_a.append(a)
            index_b.append(b)
        index_a = torch.tensor(index_a)
        index_b = torch.tensor(index_b)
        return index_a, index_b


class DiceIndexAB(nn.Module):
    def __init__(self, n_classes):
        super(DiceIndexAB, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_index(self, score, target):
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        index_a = 2 * intersect
        index_b = z_sum + y_sum
        return index_a, index_b

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        inputs = self._one_hot_encoder(inputs)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), \
            'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        index_a = []
        index_b = []
        for i in range(self.n_classes):
            a, b = self._dice_index(inputs[:, i], target[:, i])

            index_a.append(a)
            index_b.append(b)
        index_a = torch.tensor(index_a)
        index_b = torch.tensor(index_b)
        return index_a, index_b


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def train_one_epoch(args: argparse.Namespace,
                    model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    iter_num=0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 2000 // (args.batch_size * utils.get_world_size())

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        bs = len(targets)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs, losses, loss_dice, loss_ce = model(samples, targets)

        loss_dict = {
            'loss': losses.clone().detach(),
            'loss_dice': loss_dice.clone().detach(),
            'loss_ce': loss_ce.clone().detach(),
        }

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # update backfire learning rate
        if args.warmup:
            warmup = args.max_iter // args.epochs // 2
            factor = (1.0 - (iter_num - warmup) / args.max_iter) ** 0.9
            factor = min(factor, iter_num / warmup)
        else:
            factor = (1.0 - iter_num / args.max_iter) ** 0.9
        optimizer.param_groups[0]['lr'] = args.lr * factor
        optimizer.param_groups[1]['lr'] = args.lr_vit * factor
        optimizer.param_groups[2]['lr'] = args.lr_backbone * factor

        lr_dict = {
            'lr': optimizer.param_groups[0]["lr"],
        }

        iter_num += bs * utils.get_world_size()

        # log all training metrics
        metric_logger.update(**loss_dict)
        metric_logger.update(**lr_dict)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Training stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, iter_num


@torch.no_grad()
def evaluate(model, data_loader, device: torch.device, visual=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    all_dice_a, all_dice_b = [], []
    all_iou_a, all_iou_b = [], []

    device = torch.device(device)

    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        bs = len(targets)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            outputs, dice_a, dice_b, iou_a, iou_b = model(samples, targets)

        if visual:
            idx = targets[0]['id'].cpu().item()
            dice = dice_a[1:].sum() / dice_b[1:].sum()
            print('idx = ' + str(idx), 'dice = ' + str(dice.item()))
            visualize(idx=idx,
                      dice=dice.item(),
                      image=samples.tensors[0, 0].cpu().numpy(),
                      mask=outputs[0].cpu().numpy(),
                      gt=targets[0]['mask'].cpu().numpy(),
                      pts=targets[0]['points'].cpu().numpy(),
                      )

        all_dice_a.extend(utils.all_gather(dice_a))
        all_dice_b.extend(utils.all_gather(dice_b))
        all_iou_a.extend(utils.all_gather(iou_a))
        all_iou_b.extend(utils.all_gather(iou_b))

    metric_logger.synchronize_between_processes()

    # build metrics dict
    metrics = {}
    # compute mean dice
    all_dice_a = torch.vstack(all_dice_a).sum(dim=0)
    all_dice_b = torch.vstack(all_dice_b).sum(dim=0)
    all_dice = all_dice_a / all_dice_b  # finally calculate the fraction, i.e. dice index

    classes = all_dice.size(0)
    for i in range(1, classes):
        print(f"class:{i} dice={all_dice[i].item()}")
        metrics[f"class:{i}_dce"] = all_dice[i].item()
    mean_dice = all_dice[1:].mean(dim=0)  # ignore background

    # compute mean iou
    all_iou_a = torch.vstack(all_iou_a).sum(dim=0)
    all_iou_b = torch.vstack(all_iou_b).sum(dim=0)
    all_iou = all_iou_a / all_iou_b

    for i in range(1, classes):
        print(f"class:{i} iou={all_iou[i].item()}")
        metrics[f"class:{i}_iou"] = all_iou[i].item()
    miou = all_iou[1:].mean(dim=0)  # ignore background

    metrics['mean_dice'] = mean_dice.item()
    metrics['miou'] = miou.item()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()

    # SAM
    parser.add_argument('--prompt_mode', default=0, type=int, choices=[0, 1, 2, 3],
                        help="0 = train without boxes or pts, "
                             "1 = use ground-truth boxes, "
                             "2 = use ground-truth pts, "
                             "3 = use ground-truth boxes & pts.")
    parser.add_argument('--warmup', action='store_true', help='warmup at the beginning of training')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_vit', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # dataset parameters
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--dataset', type=str, choices=['rectum, word'], default='rectum')

    # runtime config
    parser.add_argument('--output_dir', default='./exp/U-SAM-Rectum',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=202307, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


class SAM(nn.Module):
    def __init__(self, args):
        super(SAM, self).__init__()

        from segment_anything import sam_model_registry
        args.model_type = 'vit_b'
        args.sam_weight = 'weight/sam_vit_b_01ec64.pth'
        self.sam = sam_model_registry[args.model_type](num_classes=args.sam_num_classes,
                                                       img_size=args.img_size,
                                                       checkpoint=args.sam_weight)

        self.pixel_mean = None
        self.pixel_std = None
        self.img_size = args.img_size
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(args.sam_num_classes)
        self.dice_index = DiceIndexAB(args.sam_num_classes)
        self.iou_index = mIoUAB(args.sam_num_classes)
        self.use_gt_box = args.use_gt_box
        self.use_gt_pts = args.use_gt_pts
        self.use_psd_box = args.use_psd_box
        self.use_psd_pts = args.use_psd_pts

        self.num_pts = 3
        assert self.num_pts in [1, 3, 5, -1]  # only support 1, 3 or 5 pts as positive prompts
        self.backbone = downsample()
        self.use_pseudo = args.use_psd_box or args.use_psd_box or args.use_psd_mask

    def forward(self, samples, targets=None):
        device = samples.tensors.device
        boxes = pts = None

        # normalization
        pixel_mean = torch.tensor(self.pixel_mean).float().to(device)
        pixel_mean = pixel_mean.unsqueeze(0).unsqueeze(0).unsqueeze(0).reshape(1, 3, 1, 1)
        pixel_std = torch.tensor(self.pixel_std).float().to(device)
        pixel_std = pixel_std.unsqueeze(0).unsqueeze(0).unsqueeze(0).reshape(1, 3, 1, 1)
        samples.tensors = (samples.tensors - pixel_mean) / pixel_std

        # use groundtruth boxes as prompt
        if self.use_gt_box or (self.use_psd_box and self.training):
            boxes = torch.vstack([targets[i]['orig_boxes'] for i in range(len(targets))])

        # use groundtruth points as prompt
        if self.use_gt_pts or (self.use_psd_pts and self.training):
            if self.num_pts == 1:
                pts_idx = torch.tensor([2]).long()
            elif self.num_pts == 3:
                pts_idx = torch.tensor([0, 2, 4]).long()
            elif self.num_pts == 5:
                pts_idx = torch.tensor([0, 1, 2, 3, 4]).long()
            else:
                pts_idx = slice(None)

            pts = torch.vstack([targets[i]['points'][:, :, pts_idx].reshape(1, -1, 2)
                                for i in range(len(targets))])
            pts_lbs = torch.ones([pts.size(0), pts.size(1)]).long()
            pts = (pts, pts_lbs)

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=pts,
                boxes=boxes,
                masks=None,
            )
        bt_feature, skip_feature = self.backbone(samples.tensors)
        image_embedding = self.sam.image_encoder(bt_feature)
        masks, low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            skip=skip_feature,
        )

        masks = self.sam.postprocess_masks(
            masks=masks,
            input_size=masks.shape[-2:],
            original_size=[self.img_size, self.img_size]
        )

        if self.training:

            def calc_loss(logits, labels, ce_loss, dice_loss, dice_weight: float = 0.6):
                loss_ce = ce_loss(logits, labels)
                loss_dice = dice_loss(logits, labels, softmax=True)
                loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
                return loss, loss_ce, loss_dice

            gt = torch.stack([targets[i]['mask'] for i in range(len(targets))], dim=0)
            sam_losses, loss_ce, loss_dice = calc_loss(masks, gt, self.ce_loss, self.dice_loss, 0.6)
            return masks, sam_losses, loss_dice, loss_ce
        else:
            masks = torch.argmax(masks, dim=1, keepdim=False)
            gt = torch.stack([targets[i]['mask'] for i in range(len(targets))], dim=0)
            dice_a, dice_b = self.dice_index(masks, gt)  # calculate numerator & denominator separately
            iou_a, iou_b = self.iou_index(masks, gt)
            return masks, dice_a, dice_b, iou_a, iou_b


def main(args):
    utils.init_distributed_mode(args)

    # prompt mode selection
    if args.prompt_mode == 0:
        args.use_gt_box = False
        args.use_gt_pts = False
        args.use_psd_box = False
        args.use_psd_pts = False
        args.use_psd_mask = False
        args.use_text = False
        prompt = 'no_prompt'
    elif args.prompt_mode == 1:
        args.use_gt_box = True
        args.use_gt_pts = False
        args.use_psd_box = False
        args.use_psd_pts = False
        args.use_psd_mask = False
        args.use_text = False
        prompt = 'gt_boxes'
    elif args.prompt_mode == 2:
        args.use_gt_box = False
        args.use_gt_pts = True
        args.use_psd_box = False
        args.use_psd_pts = False
        args.use_psd_mask = False
        args.use_text = False
        prompt = 'gt_pts'
    elif args.prompt_mode == 3:
        args.use_gt_box = True
        args.use_gt_pts = True
        args.use_psd_box = False
        args.use_psd_pts = False
        args.use_psd_mask = False
        args.use_text = False
        prompt = 'gt_boxes_pts'

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build dataset
    if args.dataset == 'rectum':
        args.sam_num_classes = 3
        args.root = '/mnt/889cdd89-1094-48ae-b221-146ffe543605/zht/datasets/Rectum/SliceData/DataV6'
        from dataset.rectum_dataloader import RectumDataloader
        dataset_train = RectumDataloader(args.root, mode='train', imgsize=(args.img_size, args.img_size))
        dataset_val = RectumDataloader(args.root, mode='test', imgsize=(args.img_size, args.img_size))

    elif args.dataset == 'word':
        args.sam_num_classes = 17
        args.root = '/mnt/889cdd89-1094-48ae-b221-146ffe543605/gwd/WORD/'
        from dataset.word_dataloader import WordDataset
        dataset_train = WordDataset(args.root, mode='train', imgsize=(args.img_size, args.img_size))
        dataset_val = WordDataset(args.root, mode='test', imgsize=(args.img_size, args.img_size))

    # build model
    model = SAM(args)

    # before eval, load checkpoint
    if args.eval:
        print('Load checkpoint')
        checkpoint_path = args.resume
        with open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
            model.load_state_dict(checkpoint['model'])

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if "image_encoder" not in n and "backbone" not in n and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if "image_encoder" in n and p.requires_grad],
            "lr": args.lr_vit,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size,
                                 sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)

    output_dir = Path(args.output_dir) / f'prompt={prompt}'

    if utils.is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    model_without_ddp.pixel_mean = pixel_mean = (0.1364736, 0.1364736, 0.1364736)
    model_without_ddp.pixel_std = pixel_std = (0.23238614, 0.23238614, 0.23238614)
    print('mean: {}, std: {}'.format(pixel_mean, pixel_std))

    if args.eval:
        print("Start evaluation")
        test_stats = evaluate(model, data_loader_val, device, visual=True)
        mean_dice = test_stats['mean_dice']
        miou = test_stats['miou']
        print('mean_dice: %.6f, miou: %.6f\n' % (mean_dice, miou))
        exit(0)

    print("Start training")
    start_time = time.time()
    best_dice = -1
    iter_num = 0
    args.max_iter = args.epochs * len(dataset_train)

    if args.resume:
        print("Resume from checkpoint", args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            iter_num = args.start_epoch * len(dataset_train)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats, iter_num = train_one_epoch(
            args, model, data_loader_train, optimizer, device, epoch, iter_num)
        test_stats = evaluate(model, data_loader_val, device)
        mean_dice = test_stats['mean_dice']
        miou = test_stats['miou']
        print('mean_dice: %.6f, miou: %.6f\n' % (mean_dice, miou))
        checkpoint_paths = []
        if mean_dice > best_dice:
            best_dice = mean_dice
            if args.output_dir:
                checkpoint_paths.append(output_dir / f'best_{mean_dice:.6f}_{miou:.6f}.pth')

        if args.output_dir:
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = parse_args()
    main(args)
