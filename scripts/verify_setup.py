import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basicsr.archs import build_network
from basicsr.data import build_dataset
from basicsr.losses import build_loss
from basicsr.utils.options import yaml_load


def unwrap_checkpoint(checkpoint):
    for key in ('model', 'state_dict', 'params', 'params_ema'):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='/home/qhm/datasets')
    parser.add_argument('--require-flickr2k', action='store_true')
    args = parser.parse_args()

    stage1 = yaml_load(str(PROJECT_ROOT / 'configs' / 'stage1_report.yaml'))
    stage2 = yaml_load(str(PROJECT_ROOT / 'configs' / 'stage2_report.yaml'))
    stage1_rep = yaml_load(str(PROJECT_ROOT / 'configs' / 'stage1_rep_report.yaml'))
    stage2_rep = yaml_load(str(PROJECT_ROOT / 'configs' / 'stage2_rep_report.yaml'))
    for options in (stage1, stage2, stage1_rep, stage2_rep):
        assert 'use_tb_logger' not in options['logger']
        assert options['logger']['wandb']['project'] == 'SPANV2'
        assert options['num_gpu'] == 1
        assert options['datasets']['train']['batch_size_per_gpu'] == 8
        assert options['train']['gradient_accumulation_steps'] == 8
        assert options['datasets']['train']['batch_size_per_gpu'] * (
            options['num_gpu']
            * options['train']['gradient_accumulation_steps']) == 64
        assert options['train']['total_iter'] == 1000000
        assert options['name'].endswith('_gb64')
    print('Training batch OK: micro-batch 8 x accumulation 8 = global batch 64')
    model = build_network(stage1['network_g'])
    checkpoint_path = PROJECT_ROOT / 'model_zoo' / 'team22_spanv2_c2.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(unwrap_checkpoint(checkpoint), strict=True)
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 16, 16))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (1, 3, 64, 64), output.shape
    assert parameters == 139104, parameters
    print(f'Official model + BasicSR registry OK: output={tuple(output.shape)}, params={parameters:,}')

    build_loss(stage1['train']['pixel_opt'])
    build_loss(stage2['train']['pixel_opt'])
    rep_model = build_network(stage1_rep['network_g']).eval()
    deployed_model = build_network(stage1['network_g']).eval()
    deployed_model.load_state_dict(rep_model.deploy_state_dict(), strict=True)
    rep_sample = torch.rand(1, 3, 13, 17)
    with torch.no_grad():
        rep_output = rep_model(rep_sample)
        deployed_output = deployed_model(rep_sample)
    assert torch.allclose(rep_output, deployed_output, rtol=1e-5, atol=2e-6)
    assert len(rep_model.state_dict()) > len(deployed_model.state_dict())
    assert sum(value.numel() for value in rep_model.deploy_state_dict().values()) == 139104
    print('BasicSR report losses, four YAML files, and REP deployment state OK')

    root = Path(args.dataset_root)
    div_hr = root / 'DIV2K' / 'HR'
    div_lr = root / 'DIV2K_bicubic' / 'LR' / 'X4'
    assert len(list(div_hr.glob('*.png'))) >= 900
    assert len(list(div_lr.glob('*.png'))) >= 900
    print('DIV2K OK: existing 900 HR/bicubic-LR pairs are reused')

    df2k_lr = root / 'DF2K' / 'LR' / 'X4'
    for index in range(1, 801):
        link = df2k_lr / f'DIV2K_{index:04d}x4.png'
        expected = div_lr / f'{index:04d}x4.png'
        assert link.is_symlink(), f'Not a symlink: {link}'
        assert link.resolve() == expected.resolve(), f'Wrong DIV2K LR target: {link}'
    print('DF2K OK: 800 DIV2K LR links point to DIV2K_bicubic/LR/X4')

    if args.require_flickr2k:
        dataset_opt = dict(stage1['datasets']['train'])
        dataset_opt['scale'] = 4
        dataset_opt['phase'] = 'train'
        dataset = build_dataset(dataset_opt)
        sample = dataset[(0, 256, 256, 0)]
        assert sample['lq'].shape == (3, 64, 64)
        assert sample['gt'].shape == (3, 256, 256)
        assert len(dataset) == 3450, len(dataset)

        stage2_opt = dict(stage2['datasets']['train'])
        stage2_opt['scale'] = 4
        stage2_opt['phase'] = 'train'
        stage2_dataset = build_dataset(stage2_opt)
        stage2_sample = stage2_dataset[0]
        assert stage2_sample['lq'].shape == (3, 128, 128)
        assert stage2_sample['gt'].shape == (3, 512, 512)
        assert len(stage2_dataset) == 3450, len(stage2_dataset)
        print(f'BasicSR DF2K OK: {len(dataset)} pairs, Stage 1/2 crop shapes verified')


if __name__ == '__main__':
    main()
