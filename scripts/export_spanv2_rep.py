#!/usr/bin/env python3
"""Fuse a SPANV2ESRRep BasicSR checkpoint into Team 22 deployment weights."""

import argparse
from pathlib import Path

import torch

from basicsr.archs import build_network


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='BasicSR REP checkpoint')
    parser.add_argument('--output', required=True, help='Output deployment .pth')
    parser.add_argument('--param-key', default='params_ema', choices=('params', 'params_ema'))
    parser.add_argument('--verify-size', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.input, map_location='cpu', weights_only=False)
    if args.param_key not in checkpoint:
        raise KeyError(f'{args.param_key!r} is not present in {args.input}')

    common = {
        'num_in_ch': 3,
        'num_out_ch': 3,
        'feature_channels': 32,
        'upscale': 4,
        'bias': False,
        'use_span_attn': False,
    }
    training_model = build_network({'type': 'SPANV2ESRRep', **common})
    training_model.load_state_dict(checkpoint[args.param_key], strict=True)
    training_model.eval()
    deploy_state = training_model.deploy_state_dict()

    deployed_model = build_network({'type': 'SPANV2ESR', **common})
    deployed_model.load_state_dict(deploy_state, strict=True)
    deployed_model.eval()

    generator = torch.Generator().manual_seed(0)
    sample = torch.rand(1, 3, args.verify_size, args.verify_size, generator=generator)
    with torch.no_grad():
        expected = training_model(sample)
        actual = deployed_model(sample)
    max_error = (expected - actual).abs().max().item()
    if not torch.allclose(expected, actual, rtol=1e-5, atol=2e-6):
        raise RuntimeError(f'Deployment fusion verification failed: max error={max_error:.3e}')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deploy_state, output)
    print(f'Saved {sum(value.numel() for value in deploy_state.values()):,} parameters to {output}')
    print(f'Max training/deployment output error: {max_error:.3e}')


if __name__ == '__main__':
    main()
