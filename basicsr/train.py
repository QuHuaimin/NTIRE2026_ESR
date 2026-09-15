import datetime
import hashlib
import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler, MultiShapeEnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_wandb_logger, make_exp_dirs, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_experiment_logger(opt, resume_state=None):
    """Initialize W&B directly when a project is configured."""
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None):
        return init_wandb_logger(opt, resume_state)
    return None


def build_data_signature(opt, micro_batches_per_epoch, dataset_size):
    """Options that must stay unchanged for an exact data-order resume."""
    dataset_opt = opt['datasets']['train']
    if dataset_opt.get('persistent_workers', False):
        raise ValueError('Exact data-order resume requires persistent_workers=false')
    meta_info_file = dataset_opt.get('meta_info_file')
    meta_info_sha256 = None
    if meta_info_file:
        with open(meta_info_file, 'rb') as file:
            meta_info_sha256 = hashlib.sha256(file.read()).hexdigest()
    accumulation_steps = int(opt['train'].get('gradient_accumulation_steps', 1))
    if accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be positive')
    if micro_batches_per_epoch % accumulation_steps:
        raise ValueError(
            'Micro-batches per epoch must be divisible by gradient_accumulation_steps')
    optimizer_steps_per_epoch = micro_batches_per_epoch // accumulation_steps
    return {
        'micro_batches_per_epoch': micro_batches_per_epoch,
        'optimizer_steps_per_epoch': optimizer_steps_per_epoch,
        'gradient_accumulation_steps': accumulation_steps,
        'effective_global_batch_size': (
            dataset_opt['batch_size_per_gpu'] * opt['world_size'] * accumulation_steps),
        'dataset_size': dataset_size,
        'dataset_type': dataset_opt['type'],
        'batch_size_per_gpu': dataset_opt['batch_size_per_gpu'],
        'num_worker_per_gpu': dataset_opt['num_worker_per_gpu'],
        'dataset_enlarge_ratio': dataset_opt.get('dataset_enlarge_ratio', 1),
        'multi_shapes': dataset_opt.get('multi_shapes'),
        'use_hflip': dataset_opt.get('use_hflip', False),
        'use_rot': dataset_opt.get('use_rot', False),
        'persistent_workers': False,
        'meta_info_sha256': meta_info_sha256,
        'scale': opt['scale'],
        'num_gpu': opt['num_gpu'],
        'world_size': opt['world_size'],
        'manual_seed': opt['manual_seed'],
    }


def resolve_resume_position(resume_state, micro_batches_per_epoch, data_signature):
    """Return the zero-based epoch and completed micro-batches in it."""
    if resume_state is None:
        return 0, 0, False

    saved_epoch = int(resume_state['epoch'])
    saved_data_state = resume_state.get('data_state')
    legacy_state = saved_data_state is None
    accumulation_steps = int(data_signature.get('gradient_accumulation_steps', 1))
    optimizer_steps_per_epoch = int(data_signature.get(
        'optimizer_steps_per_epoch', micro_batches_per_epoch // accumulation_steps))
    if legacy_state:
        if accumulation_steps != 1:
            raise RuntimeError(
                'A legacy checkpoint has no accumulation-aware data cursor and '
                'cannot be resumed with gradient_accumulation_steps > 1')
        batch_in_epoch = (
            int(resume_state['iter']) - saved_epoch * optimizer_steps_per_epoch)
    else:
        saved_signature = saved_data_state.get('signature')
        if saved_signature != data_signature:
            raise RuntimeError(
                'Training data configuration differs from the checkpoint. '
                f'Saved: {saved_signature}; current: {data_signature}')
        if int(saved_data_state['epoch']) != saved_epoch:
            raise RuntimeError('Checkpoint epoch and data cursor epoch do not match')
        if int(saved_data_state.get('iter', resume_state['iter'])) != int(resume_state['iter']):
            raise RuntimeError('Checkpoint iteration and data cursor iteration do not match')
        batch_in_epoch = int(saved_data_state['batch_in_epoch'])

    if not 0 <= batch_in_epoch <= micro_batches_per_epoch:
        raise RuntimeError(
            f'Invalid resume data cursor: epoch={saved_epoch}, '
            f'micro_batch={batch_in_epoch}, '
            f'micro_batches_per_epoch={micro_batches_per_epoch}')
    if batch_in_epoch % accumulation_steps:
        raise RuntimeError(
            'Checkpoint data cursor is inside an incomplete gradient '
            f'accumulation window: micro_batch={batch_in_epoch}, '
            f'accumulation_steps={accumulation_steps}')
    expected_iter = (saved_epoch * optimizer_steps_per_epoch
                     + batch_in_epoch // accumulation_steps)
    if not legacy_state and expected_iter != int(resume_state['iter']):
        raise RuntimeError(
            f'Checkpoint iteration {resume_state["iter"]} does not match data '
            f'cursor iteration {expected_iter}')
    if batch_in_epoch == micro_batches_per_epoch:
        return saved_epoch + 1, 0, legacy_state
    return saved_epoch, batch_in_epoch, legacy_state


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            accumulation_steps = int(
                opt['train'].get('gradient_accumulation_steps', 1))
            if accumulation_steps < 1:
                raise ValueError('gradient_accumulation_steps must be positive')
            train_set = build_dataset(dataset_opt)
            if dataset_opt.get('multi_shapes'):
                train_sampler = MultiShapeEnlargedSampler(
                    train_set,
                    opt['world_size'],
                    opt['rank'],
                    dataset_enlarge_ratio,
                    dataset_opt['batch_size_per_gpu'],
                    dataset_opt['multi_shapes'],
                    dataset_opt.get('use_rot', True),
                    accumulation_steps)
            else:
                train_sampler = EnlargedSampler(
                    train_set,
                    opt['world_size'],
                    opt['rank'],
                    dataset_enlarge_ratio,
                    dataset_opt['batch_size_per_gpu'],
                    accumulation_steps)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_micro_batches_per_epoch = len(train_loader)
            if num_micro_batches_per_epoch % accumulation_steps:
                raise RuntimeError(
                    'Sampler produced an incomplete gradient accumulation window')
            num_optimizer_steps_per_epoch = (
                num_micro_batches_per_epoch // accumulation_steps)
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_optimizer_steps_per_epoch)
            effective_global_batch_size = (
                dataset_opt['batch_size_per_gpu'] * opt['world_size']
                * accumulation_steps)
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tMicro-batch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tGradient accumulation steps: {accumulation_steps}'
                        f'\n\tEffective global batch size: {effective_global_batch_size}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tMicro-batches per epoch: {num_micro_batches_per_epoch}'
                        f'\n\tOptimizer steps per epoch: {num_optimizer_steps_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return (train_loader, train_sampler, val_loaders, total_epochs, total_iters,
            num_micro_batches_per_epoch)


def load_resume_state(opt):
    resume_state_path = None
    requested_iter = opt.get('resume_iter')
    if requested_iter is not None:
        if requested_iter < 0:
            raise ValueError('--resume_iter must be zero or greater')
        iteration = str(requested_iter)
        resume_state_path = osp.join(
            opt['path']['training_states'], f'{iteration}.state')
        model_path = osp.join(opt['path']['models'], f'net_g_{iteration}.pth')
        missing = [path for path in (resume_state_path, model_path) if not osp.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f'Iteration {requested_iter} is not a complete checkpoint; '
                f'missing: {missing}')
        opt['path']['resume_state'] = resume_state_path
    elif opt['auto_resume']:
        state_path = opt['path']['training_states']
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                for state_iter in sorted(states, reverse=True):
                    iteration = f'{state_iter:.0f}'
                    model_path = osp.join(opt['path']['models'], f'net_g_{iteration}.pth')
                    if osp.isfile(model_path):
                        resume_state_path = osp.join(state_path, f'{iteration}.state')
                        break
                if resume_state_path is None:
                    raise RuntimeError(
                        f'Found training states in {state_path}, but none has a '
                        'matching net_g checkpoint')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            resume_state_path,
            map_location=lambda storage, loc: storage.cuda(device_id),
            weights_only=False)
        state_name = osp.basename(resume_state_path)
        state_stem = state_name[:-6] if state_name.endswith('.state') else state_name
        try:
            filename_iter = int(float(state_stem))
        except ValueError:
            filename_iter = None
        if filename_iter is not None and filename_iter != int(resume_state['iter']):
            raise RuntimeError(
                f'Resume state filename iteration {filename_iter} does not match '
                f'its saved iteration {resume_state["iter"]}')
        check_resume(opt, resume_state['iter'])
        model_path = opt['path'].get('pretrain_network_g')
        if model_path is None or not osp.isfile(model_path):
            raise RuntimeError(
                f'Resume state {resume_state_path} has no matching model checkpoint '
                f'for iteration {resume_state["iter"]}')
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    wandb_logger = init_experiment_logger(opt, resume_state)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    (train_loader, train_sampler, val_loaders, total_epochs, total_iters,
     num_micro_batches_per_epoch) = result
    accumulation_steps = int(opt['train'].get('gradient_accumulation_steps', 1))
    data_signature = build_data_signature(
        opt, num_micro_batches_per_epoch, len(train_loader.dataset))

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch, resume_batch, legacy_data_state = resolve_resume_position(
            resume_state, num_micro_batches_per_epoch, data_signature)
        current_iter = resume_state['iter']
        if legacy_data_state:
            logger.warning(
                'Legacy checkpoint has no explicit data cursor; derived '
                f'epoch={start_epoch}, batch={resume_batch} from iter={current_iter}.')
    else:
        start_epoch = 0
        resume_batch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, wandb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()
    training_complete = False

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        batch_in_epoch = 0

        if epoch == start_epoch and resume_batch:
            logger.info(
                f'Fast-forwarding dataloader by '
                f'{resume_batch}/{num_micro_batches_per_epoch} completed '
                'micro-batches to restore the exact checkpoint cursor.')
            for skipped in range(resume_batch):
                if train_data is None:
                    raise RuntimeError('Dataloader ended while restoring the checkpoint cursor')
                train_data = prefetcher.next()
                if (skipped + 1) % 500 == 0 or skipped + 1 == resume_batch:
                    logger.info(f'Dataloader resume progress: {skipped + 1}/{resume_batch}')
            batch_in_epoch = resume_batch
        if resume_state is not None and epoch == start_epoch:
            model.restore_training_rng_state()
            resume_batch = 0
            msg_logger.reset_start_time()
            data_timer.start()
            iter_timer.start()

        while train_data is not None:
            if current_iter >= total_iters:
                training_complete = True
                break
            data_timer.record()

            accumulation_step = batch_in_epoch % accumulation_steps
            next_iter = current_iter + 1
            if accumulation_step == 0:
                model.update_learning_rate(
                    next_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(
                next_iter,
                accumulation_step=accumulation_step,
                accumulation_steps=accumulation_steps)
            batch_in_epoch += 1
            train_data = prefetcher.next()

            if accumulation_step + 1 < accumulation_steps:
                continue

            current_iter = next_iter
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.set_training_wandb_run_id(
                    opt['logger'].get('wandb', {}).get('active_run_id'))
                model.set_training_data_state({
                    'epoch': epoch,
                    'iter': current_iter,
                    'batch_in_epoch': batch_in_epoch,
                    'signature': data_signature,
                })
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, wandb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
        # end of iter

        if batch_in_epoch % accumulation_steps:
            raise RuntimeError(
                'Dataloader ended inside a gradient accumulation window')
        if training_complete:
            break

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, wandb_logger, opt['val']['save_img'])
    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
