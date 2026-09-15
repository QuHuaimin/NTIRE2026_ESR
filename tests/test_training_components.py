import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.data import build_dataset
from basicsr.data.data_sampler import MultiShapeEnlargedSampler
from basicsr.losses import build_loss
from basicsr.losses.report_loss import _fft_distance
from basicsr.models.lr_scheduler import CosineAnnealingRestartLR
from basicsr.models.sr_model import SRModel, calculate_output_gradient_diagnostics
from basicsr.train import resolve_resume_position
from basicsr.utils.logger import MessageLogger, init_wandb_logger


class TrainingComponentsTest(unittest.TestCase):
    def test_exact_resume_cursor(self):
        signature = {
            'micro_batches_per_epoch': 43128,
            'optimizer_steps_per_epoch': 5391,
            'gradient_accumulation_steps': 8,
            'effective_global_batch_size': 64,
            'dataset_size': 3450,
            'dataset_type': 'ReportMultiShapePairedImageDataset',
            'batch_size_per_gpu': 8,
            'num_worker_per_gpu': 8,
            'dataset_enlarge_ratio': 100,
            'multi_shapes': [[256, 256]],
            'use_hflip': True,
            'use_rot': True,
            'persistent_workers': False,
            'meta_info_sha256': 'test',
            'scale': 4,
            'num_gpu': 1,
            'world_size': 1,
            'manual_seed': 10,
        }
        legacy = {'epoch': 15, 'iter': 650000}
        self.assertEqual(
            resolve_resume_position(
                legacy, 43125,
                {'batch_size_per_gpu': 8, 'world_size': 1}),
            (15, 3125, True))

        current = {
            'epoch': 120,
            'iter': 650000,
            'data_state': {
                'epoch': 120,
                'iter': 650000,
                'batch_in_epoch': 24640,
                'signature': signature,
            },
        }
        self.assertEqual(
            resolve_resume_position(current, 43128, signature),
            (120, 24640, False))

        changed = dict(signature, batch_size_per_gpu=4)
        with self.assertRaises(RuntimeError):
            resolve_resume_position(current, 43128, changed)

        mismatched_iter = dict(current)
        mismatched_iter['data_state'] = dict(current['data_state'], iter=649999)
        with self.assertRaises(RuntimeError):
            resolve_resume_position(mismatched_iter, 43128, signature)

        mismatched_batch = dict(current)
        mismatched_batch['data_state'] = dict(
            current['data_state'], batch_in_epoch=24639)
        with self.assertRaises(RuntimeError):
            resolve_resume_position(mismatched_batch, 43128, signature)

        with self.assertRaises(RuntimeError):
            resolve_resume_position(legacy, 43128, signature)

    def test_gradient_accumulation_matches_full_batch(self):
        def make_model(state_dict=None):
            model = SRModel.__new__(SRModel)
            model.net_g = torch.nn.Conv2d(3, 3, 1, bias=False)
            if state_dict is not None:
                model.net_g.load_state_dict(state_dict)
            model.optimizer_g = torch.optim.SGD(model.net_g.parameters(), lr=0.1)
            model.cri_pix = torch.nn.L1Loss()
            model.cri_perceptual = None
            model.gradient_diagnostics_interval = 0
            model.ema_decay = 0
            model.reduce_loss_dict = lambda losses: losses
            return model

        torch.manual_seed(7)
        full_batch_model = make_model()
        accumulated_model = make_model(full_batch_model.net_g.state_dict())
        lq = torch.rand(4, 3, 5, 5)
        gt = torch.rand(4, 3, 5, 5)

        full_batch_model.lq = lq
        full_batch_model.gt = gt
        full_batch_model.optimize_parameters(1)

        initial = {
            name: value.detach().clone()
            for name, value in accumulated_model.net_g.state_dict().items()
        }
        for accumulation_step in range(2):
            start = accumulation_step * 2
            accumulated_model.lq = lq[start:start + 2]
            accumulated_model.gt = gt[start:start + 2]
            accumulated_model.optimize_parameters(
                1, accumulation_step=accumulation_step, accumulation_steps=2)
            if accumulation_step == 0:
                for name, value in accumulated_model.net_g.state_dict().items():
                    torch.testing.assert_close(value, initial[name])

        for full, accumulated in zip(
                full_batch_model.net_g.parameters(),
                accumulated_model.net_g.parameters()):
            torch.testing.assert_close(full, accumulated, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(
            full_batch_model.log_dict['l_pix'],
            accumulated_model.log_dict['l_pix'])

    def test_native_wandb_metrics(self):
        class FakeRun:
            def __init__(self):
                self.records = []

            def log(self, values):
                self.records.append(values)

        run = FakeRun()
        logger = MessageLogger({
            'name': 'wandb-test',
            'logger': {'print_freq': 1},
            'train': {'total_iter': 100},
        }, start_iter=0, wandb_logger=run)
        logger({
            'epoch': 2,
            'iter': 10,
            'lrs': [1e-3],
            'time': 0.2,
            'data_time': 0.1,
            'l_pix': 0.25,
            'grad_fft_to_pixel_ratio': 0.08,
        })
        self.assertEqual(len(run.records), 1)
        self.assertEqual(run.records[0]['iteration'], 10)
        self.assertEqual(run.records[0]['train/epoch'], 2)
        self.assertEqual(run.records[0]['train/learning_rate'], 1e-3)
        self.assertEqual(run.records[0]['train/losses/l_pix'], 0.25)
        self.assertEqual(
            run.records[0]['train/gradient_diagnostics/grad_fft_to_pixel_ratio'],
            0.08)

    def test_wandb_resume_uses_checkpoint_run(self):
        calls = []

        class FakeRun:
            id = 'checkpoint-run'

            def define_metric(self, *args, **kwargs):
                pass

        fake_wandb = SimpleNamespace(
            util=SimpleNamespace(generate_id=lambda: 'generated-run'),
            init=lambda **kwargs: calls.append(kwargs) or FakeRun())
        with tempfile.TemporaryDirectory() as temp:
            run_id_path = Path(temp) / 'wandb_run_id.txt'
            run_id_path.write_text('stale-run\n', encoding='utf-8')
            opt = {
                'name': 'wandb-resume-test',
                'path': {'experiments_root': temp},
                'logger': {'wandb': {'project': 'SPANV2', 'resume_id': None}},
            }
            with patch.dict(sys.modules, {'wandb': fake_wandb}):
                init_wandb_logger(opt, {'wandb_run_id': 'checkpoint-run'})

            self.assertEqual(calls[0]['id'], 'checkpoint-run')
            self.assertEqual(calls[0]['resume'], 'must')
            self.assertEqual(run_id_path.read_text(encoding='utf-8').strip(), 'checkpoint-run')

    def test_wandb_rewind_uses_last_matching_internal_step(self):
        calls = []

        class FakeRun:
            def define_metric(self, *args, **kwargs):
                pass

        class FakeApiRun:
            def scan_history(self, keys):
                return iter([
                    {'_step': 6563, 'iteration': 650000},
                    {'_step': 6564, 'iteration': 650000},
                    {'_step': 6565, 'iteration': 650100},
                ])

        fake_wandb = SimpleNamespace(
            Api=lambda: SimpleNamespace(run=lambda path: FakeApiRun()),
            util=SimpleNamespace(generate_id=lambda: 'generated-run'),
            init=lambda **kwargs: calls.append(kwargs) or FakeRun())
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, 'wandb_run_id.txt').write_text('source-run\n', encoding='utf-8')
            opt = {
                'name': 'wandb-rewind-test',
                'path': {'experiments_root': temp},
                'logger': {'wandb': {
                    'project': 'SPANV2',
                    'resume_id': None,
                    'resume_mode': 'rewind',
                }},
            }
            with patch.dict(sys.modules, {'wandb': fake_wandb}):
                init_wandb_logger(opt, {'iter': 650000})

            self.assertEqual(calls[0]['resume_from'], 'source-run?_step=6564')
            self.assertNotIn('resume', calls[0])
            self.assertEqual(opt['logger']['wandb']['active_run_id'], 'source-run')

    def test_official_model_registry(self):
        model = build_network({
            'type': 'SPANV2ESR',
            'num_in_ch': 3,
            'num_out_ch': 3,
            'feature_channels': 32,
            'upscale': 4,
            'bias': False,
            'use_span_attn': False,
        })
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 139104)
        self.assertEqual(tuple(model(torch.rand(1, 3, 8, 8)).shape), (1, 3, 32, 32))

    def test_rep_model_fuses_to_official_topology(self):
        common = {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'feature_channels': 32,
            'upscale': 4,
            'bias': False,
            'use_span_attn': False,
        }
        rep_model = build_network({'type': 'SPANV2ESRRep', **common}).eval()
        deployed_model = build_network({'type': 'SPANV2ESR', **common}).eval()
        deployed_state = rep_model.deploy_state_dict()
        deployed_model.load_state_dict(deployed_state, strict=True)

        sample = torch.rand(1, 3, 13, 17)
        with torch.no_grad():
            rep_output = rep_model(sample)
            deployed_output = deployed_model(sample)
        self.assertTrue(torch.allclose(rep_output, deployed_output, rtol=1e-5, atol=2e-6))
        self.assertEqual(sum(value.numel() for value in deployed_state.values()), 139104)
        self.assertGreater(
            sum(parameter.numel() for parameter in rep_model.parameters()), 139104)
        rep_model.train()
        rep_model(torch.rand(1, 3, 8, 8)).mean().backward()
        self.assertTrue(all(
            parameter.grad is not None for parameter in rep_model.parameters()))

    def test_pairing_crop_and_batch_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hr_dir, lr_dir = root / 'HR', root / 'LR'
            hr_dir.mkdir()
            lr_dir.mkdir()
            rng = np.random.default_rng(1)
            lr = rng.integers(0, 256, size=(80, 96, 3), dtype=np.uint8)
            hr = np.repeat(np.repeat(lr, 4, axis=0), 4, axis=1)
            Image.fromarray(lr).save(lr_dir / '0001x4.png')
            Image.fromarray(hr).save(hr_dir / '0001.png')
            meta = root / 'meta.txt'
            meta.write_text('0001.png (320,384,3)\n', encoding='utf-8')
            dataset = build_dataset({
                'name': 'test',
                'type': 'ReportMultiShapePairedImageDataset',
                'dataroot_gt': str(hr_dir),
                'dataroot_lq': str(lr_dir),
                'meta_info_file': str(meta),
                'filename_tmpl': '{}x4',
                'io_backend': {'type': 'disk'},
                'scale': 4,
                'multi_shapes': [[256, 320]],
                'use_hflip': True,
                'use_rot': True,
            })
            sample = dataset[(0, 256, 320, 1)]
            self.assertEqual(tuple(sample['lq'].shape), (3, 80, 64))
            self.assertEqual(tuple(sample['gt'].shape), (3, 320, 256))

            sampler = MultiShapeEnlargedSampler(
                dataset, 1, 0, ratio=8, batch_size=4,
                shapes=[[256, 320], [320, 256]], use_rot=True,
                accumulation_steps=2)
            requests = list(iter(sampler))
            self.assertEqual(len(requests) % 8, 0)
            for start in range(0, len(requests), 8):
                self.assertEqual(
                    len({item[1:] for item in requests[start:start + 8]}), 1)

    def test_report_losses_and_scheduler(self):
        prediction = torch.zeros(1, 3, 16, 16, requires_grad=True)
        target = torch.ones_like(prediction)
        stage1_loss = build_loss({
            'type': 'L1FFTReportLoss', 'l1_weight': 1.0, 'fft_weight': 0.05,
            'fft_norm': 'ortho'})
        safmn_stage1_loss = build_loss({
            'type': 'L1FFTReportLoss', 'l1_weight': 1.0, 'fft_weight': 0.05,
            'fft_norm': 'backward'})
        stage2_loss = build_loss({
            'type': 'MSEGradientReportLoss', 'mse_weight': 5.0, 'gradient_weight': 3.0})
        total = stage1_loss(prediction, target) + stage2_loss(prediction, target)
        total.backward()
        self.assertIsNotNone(prediction.grad)

        stage1_components = stage1_loss.loss_components(prediction, target)
        stage2_components = stage2_loss.loss_components(prediction, target)
        torch.testing.assert_close(sum(stage1_components.values()), stage1_loss(prediction, target))
        torch.testing.assert_close(sum(stage2_components.values()), stage2_loss(prediction, target))
        stage1_diagnostics = calculate_output_gradient_diagnostics(
            stage1_loss, prediction, target)
        stage2_diagnostics = calculate_output_gradient_diagnostics(
            stage2_loss, prediction, target)
        self.assertEqual(set(stage1_diagnostics), {
            'grad_pixel_l2', 'grad_fft_l2', 'grad_fft_to_pixel_ratio',
            'grad_pixel_fft_cosine'})
        self.assertEqual(set(stage2_diagnostics), {
            'grad_mse_l2', 'grad_gradient_l2', 'grad_gradient_to_mse_ratio',
            'grad_mse_gradient_cosine'})
        for diagnostics in (stage1_diagnostics, stage2_diagnostics):
            self.assertTrue(all(torch.isfinite(value) for value in diagnostics.values()))
            cosine = next(
                value for name, value in diagnostics.items() if name.endswith('_cosine'))
            self.assertLessEqual(abs(cosine.item()), 1.0)

        fft_prediction = torch.rand(2, 3, 12, 16)
        fft_target = torch.rand_like(fft_prediction)
        pred_fft = torch.fft.rfft2(fft_prediction, norm='ortho')
        target_fft = torch.fft.rfft2(fft_target, norm='ortho')
        expected_fft = F.l1_loss(
            torch.stack((pred_fft.real, pred_fft.imag), dim=-1),
            torch.stack((target_fft.real, target_fft.imag), dim=-1))
        torch.testing.assert_close(
            _fft_distance(fft_prediction, fft_target, norm='ortho'), expected_fft)

        safmn_pred_fft = torch.fft.rfft2(fft_prediction)
        safmn_target_fft = torch.fft.rfft2(fft_target)
        expected_safmn_fft = F.l1_loss(
            torch.stack((safmn_pred_fft.real, safmn_pred_fft.imag), dim=-1),
            torch.stack((safmn_target_fft.real, safmn_target_fft.imag), dim=-1))
        torch.testing.assert_close(
            _fft_distance(fft_prediction, fft_target, norm='backward'),
            expected_safmn_fft)
        torch.testing.assert_close(
            safmn_stage1_loss.loss_components(fft_prediction, fft_target)['fft'],
            0.05 * expected_safmn_fft)

        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=5e-4)
        scheduler = CosineAnnealingRestartLR(
            optimizer, periods=[600000, 400000], restart_weights=[1, 0.5], eta_min=1e-6)
        scheduler.last_epoch = 600001
        self.assertAlmostEqual(scheduler.get_lr()[0], 2.505e-4)
        scheduler.last_epoch = 1000000
        self.assertAlmostEqual(scheduler.get_lr()[0], 1e-6)


if __name__ == '__main__':
    unittest.main()
