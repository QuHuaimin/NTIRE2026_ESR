"""Paired BasicSR dataset for report-aligned multi-shape training."""

import random

import cv2
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def _paired_rectangular_crop(img_gt, img_lq, gt_height, gt_width, scale, gt_path):
    h_lq, w_lq = img_lq.shape[:2]
    h_gt, w_gt = img_gt.shape[:2]
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatch for {gt_path}: GT {(h_gt, w_gt)}, LQ {(h_lq, w_lq)}, scale {scale}')

    lq_height, lq_width = gt_height // scale, gt_width // scale
    if h_lq < lq_height or w_lq < lq_width:
        raise ValueError(
            f'LQ image {gt_path} is {(h_lq, w_lq)}, smaller than requested '
            f'crop {(lq_height, lq_width)}')

    top = random.randint(0, h_lq - lq_height)
    left = random.randint(0, w_lq - lq_width)
    img_lq = img_lq[top:top + lq_height, left:left + lq_width, ...]
    img_gt = img_gt[top * scale:(top + lq_height) * scale,
                    left * scale:(left + lq_width) * scale, ...]
    return img_gt, img_lq


def _augment_pair(img_gt, img_lq, use_hflip, rotation):
    if use_hflip and random.random() < 0.5:
        img_gt = cv2.flip(img_gt, 1)
        img_lq = cv2.flip(img_lq, 1)
    if rotation:
        img_gt = np.ascontiguousarray(np.rot90(img_gt, rotation))
        img_lq = np.ascontiguousarray(np.rot90(img_lq, rotation))
    return img_gt, img_lq


@DATASET_REGISTRY.register()
class ReportMultiShapePairedImageDataset(data.Dataset):
    """BasicSR paired dataset accepting shape-aware indices from the sampler."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif opt.get('meta_info_file') is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, request):
        if isinstance(request, (tuple, list)):
            index, gt_height, gt_width, rotation = request
        else:
            index = request
            gt_height, gt_width = random.choice(self.opt['multi_shapes'])
            rotation = random.randrange(4) if self.opt.get('use_rot', False) else 0

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
        img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)
        img_gt, img_lq = _paired_rectangular_crop(
            img_gt, img_lq, gt_height, gt_width, self.opt['scale'], gt_path)
        img_gt, img_lq = _augment_pair(
            img_gt, img_lq, self.opt.get('use_hflip', False), rotation)

        if self.opt.get('color') == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
