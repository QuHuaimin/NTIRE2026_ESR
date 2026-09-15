import datetime
import logging
import time
from pathlib import Path

from .dist_util import get_dist_info, master_only

initialized_logger = {}


class AvgTimer():

    def __init__(self, window=200):
        self.window = window  # average window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        self.start_time = self.tic = time.time()

    def record(self):
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time


class MessageLogger():
    """Write formatted console logs and native Weights & Biases metrics.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
        start_iter (int): Start iter. Default: 1.
        wandb_logger: Initialized W&B run. Default: None.
    """

    def __init__(self, opt, start_iter=1, wandb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.wandb_logger = wandb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def reset_start_time(self):
        self.start_time = time.time()

    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')
        wandb_metrics = {'iteration': current_iter, 'train/epoch': epoch}

        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:(')
        for index, v in enumerate(lrs):
            message += f'{v:.3e},'
            name = 'train/learning_rate' if index == 0 else f'train/learning_rate_{index}'
            wandb_metrics[name] = v
        message += ')] '

        # time and estimated time
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '
            wandb_metrics['train/iteration_time'] = iter_time
            wandb_metrics['train/data_time'] = data_time

        # other items, especially losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
            if k.startswith('l_'):
                namespace = 'train/losses'
            elif k.startswith('grad_'):
                namespace = 'train/gradient_diagnostics'
            else:
                namespace = 'train'
            wandb_metrics[f'{namespace}/{k}'] = v
        if self.wandb_logger is not None:
            self.wandb_logger.log(wandb_metrics)
        self.logger.info(message)


def _find_wandb_step(wandb, project, entity, run_id, iteration):
    """Find the last W&B history row belonging to a training iteration."""
    run_path = f'{entity}/{project}/{run_id}' if entity else f'{project}/{run_id}'
    api_run = wandb.Api().run(run_path)
    matching_steps = [
        int(row['_step'])
        for row in api_run.scan_history(keys=['_step', 'iteration'])
        if row.get('_step') is not None and row.get('iteration') is not None
        and int(row['iteration']) == int(iteration)
    ]
    if not matching_steps:
        raise RuntimeError(
            f'W&B run {run_id} has no history row for training iteration '
            f'{iteration}; refusing to rewind an uncertain position')
    return max(matching_steps)


@master_only
def init_wandb_logger(opt, resume_state=None):
    """Initialize W&B with resume, destructive rewind, or branch semantics."""
    import wandb
    logger = get_root_logger()

    wandb_opt = opt['logger']['wandb']
    project = wandb_opt['project']
    entity = wandb_opt.get('entity')
    run_id_path = Path(opt['path']['experiments_root']) / 'wandb_run_id.txt'
    checkpoint_id = resume_state.get('wandb_run_id') if resume_state else None
    configured_id = wandb_opt.get('resume_id')
    persisted_id = run_id_path.read_text(encoding='utf-8').strip() if run_id_path.is_file() else None
    if checkpoint_id:
        source_id = checkpoint_id
        id_source = 'checkpoint'
        if configured_id and configured_id != checkpoint_id:
            logger.warning(
                f'Ignore configured W&B run ID {configured_id}; resumed checkpoint '
                f'is bound to {checkpoint_id}.')
        if persisted_id and persisted_id != checkpoint_id:
            logger.warning(
                f'Replace stale W&B run ID {persisted_id} with checkpoint run ID '
                f'{checkpoint_id}.')
    elif configured_id:
        source_id = configured_id
        id_source = 'config'
    elif persisted_id:
        source_id = persisted_id
        id_source = 'file'
    else:
        source_id = None
        id_source = 'generated'

    requested_mode = wandb_opt.get('resume_mode', 'resume')
    init_kwargs = {
        'name': opt['name'],
        'config': opt,
        'project': project,
        'entity': entity,
        'group': wandb_opt.get('group'),
        'tags': wandb_opt.get('tags'),
        'dir': opt['path']['experiments_root'],
    }
    if resume_state is not None and requested_mode in ('rewind', 'fork'):
        if source_id is None:
            raise RuntimeError(f'W&B {requested_mode} requires an existing run ID')
        rewind_step = _find_wandb_step(
            wandb, project, entity, source_id, resume_state['iter'])
        run_moment = f'{source_id}?_step={rewind_step}'
        if requested_mode == 'rewind':
            wandb_id = source_id
            init_kwargs['resume_from'] = run_moment
            active_mode = f'rewind@{rewind_step}'
        else:
            wandb_id = wandb.util.generate_id()
            init_kwargs['id'] = wandb_id
            init_kwargs['fork_from'] = run_moment
            active_mode = f'fork@{rewind_step}'
    else:
        wandb_id = source_id or wandb.util.generate_id()
        init_kwargs['id'] = wandb_id
        init_kwargs['resume'] = (
            'must' if resume_state is not None and source_id is not None else 'allow')
        active_mode = init_kwargs['resume']

    wandb_opt['active_run_id'] = wandb_id
    try:
        run = wandb.init(**init_kwargs)
    except Exception as error:
        if resume_state is not None and requested_mode in ('rewind', 'fork'):
            raise RuntimeError(
                f'W&B {requested_mode} could not be initialized. This feature '
                'may require account-side preview access; use '
                '--wandb_resume_mode resume or contact W&B support to enable it.') from error
        raise
    run_id_path.write_text(wandb_id + '\n', encoding='utf-8')
    run.define_metric('iteration')
    run.define_metric('train/*', step_metric='iteration')
    run.define_metric('train/gradient_diagnostics/*', step_metric='iteration')
    run.define_metric('validation/*', step_metric='iteration')
    logger.info(
        f'Use native wandb logger with id={wandb_id}; project={project}; '
        f'mode={active_mode}; id_source={id_source}.')
    return run


def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    from basicsr.version import __version__
    msg = r"""
                ____                _       _____  ____
               / __ ) ____ _ _____ (_)_____/ ___/ / __ \
              / __  |/ __ `// ___// // ___/\__ \ / /_/ /
             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
            /_____/ \__,_//____//_/ \___//____//_/ |_|
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: '
            f'\n\tBasicSR: {__version__}'
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg
