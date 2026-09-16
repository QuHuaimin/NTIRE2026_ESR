#!/usr/bin/env python3
"""Compare plausible interpretations of SPANV2's unpublished loss formulas."""

import argparse
from pathlib import Path

import cv2
import torch
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.losses.report_loss import L1FFTReportLoss, MSEGradientReportLoss
from basicsr.models.sr_model import calculate_output_gradient_diagnostics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--param-key', default='params_ema', choices=('params', 'params_ema'))
    parser.add_argument('--data-root', default='~/datasets/DF2K')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_pair(root, name, crop_size):
    stem = Path(name).stem
    gt = cv2.imread(str(root / 'HR' / name), cv2.IMREAD_COLOR)
    lq = cv2.imread(str(root / 'LR' / 'X4' / f'{stem}x4.png'), cv2.IMREAD_COLOR)
    if gt is None or lq is None:
        raise FileNotFoundError(f'Could not load pair for {name}')
    height = min(crop_size, gt.shape[0] // 4 * 4)
    width = min(crop_size, gt.shape[1] // 4 * 4)
    gt = gt[:height, :width, ::-1].copy()
    lq = lq[:height // 4, :width // 4, ::-1].copy()
    tensors = []
    for image in (lq, gt):
        tensors.append(
            torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div_(255))
    return tensors


def loss_values(prediction, target):
    pred_fft = torch.fft.rfft2(prediction, norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')
    pred_fft_parts = torch.stack((pred_fft.real, pred_fft.imag), dim=-1)
    target_fft_parts = torch.stack((target_fft.real, target_fft.imag), dim=-1)
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return {
        'l1': F.l1_loss(prediction, target),
        'mse': F.mse_loss(prediction, target),
        'fft_complex_ortho': torch.abs(pred_fft - target_fft).mean(),
        'fft_real_imag_l1_ortho': F.l1_loss(pred_fft_parts, target_fft_parts),
        'fft_amplitude_ortho': F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft)),
        'gradient_l1_sum': F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy),
        'gradient_l1_mean': (
            F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)) / 2,
        'gradient_mse_sum': F.mse_loss(pred_dx, target_dx) + F.mse_loss(pred_dy, target_dy),
        'gradient_mse_mean': (
            F.mse_loss(pred_dx, target_dx) + F.mse_loss(pred_dy, target_dy)) / 2,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = build_network({
        'type': 'SPANV2ESR',
        'num_in_ch': 3,
        'num_out_ch': 3,
        'feature_channels': 32,
        'upscale': 4,
        'bias': False,
        'use_span_attn': False,
    }).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint[args.param_key], strict=True)
    stage1_loss = L1FFTReportLoss().to(device)
    stage2_loss = MSEGradientReportLoss().to(device)

    root = Path(args.data_root).expanduser()
    names = [line.split()[0] for line in (root / 'meta_info_DF2K.txt').read_text().splitlines()]
    names = names[:args.samples]
    totals = {}
    for name in names:
        lq, gt = (tensor.to(device) for tensor in load_pair(root, name, args.crop_size))
        with torch.no_grad():
            prediction = model(lq)
        values = loss_values(prediction, gt)
        values.update(calculate_output_gradient_diagnostics(
            stage1_loss, prediction, gt))
        values.update(calculate_output_gradient_diagnostics(
            stage2_loss, prediction, gt))
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value.item()

    means = {key: value / len(names) for key, value in totals.items()}
    for key, value in means.items():
        print(f'{key:24s} {value:.8f}')
    stage1 = means['l1'] + 0.05 * means['fft_real_imag_l1_ortho']
    stage2_l1 = 5 * means['mse'] + 3 * means['gradient_l1_sum']
    stage2_mse = 5 * means['mse'] + 3 * means['gradient_mse_mean']
    print(f'current_stage1_total     {stage1:.8f}')
    print(f'current_stage2_total     {stage2_l1:.8f}')
    print(f'mse_gradient_candidate   {stage2_mse:.8f}')


if __name__ == '__main__':
    main()
