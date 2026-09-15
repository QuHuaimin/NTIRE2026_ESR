"""Loss combinations reported by XiaomiMM for SPANV2 training."""

from collections import OrderedDict

import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


def _fft_distance(prediction, target, norm='ortho'):
    if norm not in ('backward', 'forward', 'ortho'):
        raise ValueError(f'Unsupported FFT normalization: {norm}')
    if norm == 'backward':
        pred_fft = torch.fft.rfft2(prediction)
        target_fft = torch.fft.rfft2(target)
    else:
        pred_fft = torch.fft.rfft2(prediction, norm=norm)
        target_fft = torch.fft.rfft2(target, norm=norm)
    pred_fft = torch.stack((pred_fft.real, pred_fft.imag), dim=-1)
    target_fft = torch.stack((target_fft.real, target_fft.imag), dim=-1)
    return F.l1_loss(pred_fft, target_fft)


def _gradient_distance(prediction, target):
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


@LOSS_REGISTRY.register()
class L1FFTReportLoss(nn.Module):
    """Stage 1 loss: spatial L1 plus real/imaginary FFT L1.

    The report publishes the weights but not the FFT-loss formula. ``fft_norm``
    makes the uncertain normalization explicit: ``ortho`` is the stable-scale
    baseline, while ``backward`` exactly matches SAFMN's default ``rfft2`` scale.
    """

    def __init__(self, l1_weight=1.0, fft_weight=0.05, fft_norm='ortho',
                 reduction='mean'):
        super().__init__()
        if reduction != 'mean':
            raise ValueError('L1FFTReportLoss currently supports reduction=mean only')
        self.l1_weight = l1_weight
        self.fft_weight = fft_weight
        if fft_norm not in ('backward', 'forward', 'ortho'):
            raise ValueError(f'Unsupported FFT normalization: {fft_norm}')
        self.fft_norm = fft_norm

    def loss_components(self, prediction, target, weight=None):
        if weight is not None:
            raise ValueError('Per-pixel weights are not supported by L1FFTReportLoss')
        return OrderedDict([
            ('pixel', self.l1_weight * F.l1_loss(prediction, target)),
            ('fft', self.fft_weight * _fft_distance(
                prediction, target, norm=self.fft_norm)),
        ])

    def forward(self, prediction, target, weight=None, **kwargs):
        return sum(self.loss_components(prediction, target, weight).values())


@LOSS_REGISTRY.register()
class MSEGradientReportLoss(nn.Module):
    """Stage 2 loss: weighted MSE plus first-order gradient distance."""

    def __init__(self, mse_weight=5.0, gradient_weight=3.0, reduction='mean'):
        super().__init__()
        if reduction != 'mean':
            raise ValueError('MSEGradientReportLoss currently supports reduction=mean only')
        self.mse_weight = mse_weight
        self.gradient_weight = gradient_weight

    def loss_components(self, prediction, target, weight=None):
        if weight is not None:
            raise ValueError('Per-pixel weights are not supported by MSEGradientReportLoss')
        return OrderedDict([
            ('mse', self.mse_weight * F.mse_loss(prediction, target)),
            ('gradient', self.gradient_weight * _gradient_distance(prediction, target)),
        ])

    def forward(self, prediction, target, weight=None, **kwargs):
        return sum(self.loss_components(prediction, target, weight).values())
