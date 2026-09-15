import math
import torch
from torch.utils.data.sampler import Sampler


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1, batch_size=1,
                 accumulation_steps=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.effective_batch_size = batch_size * accumulation_steps
        target = len(self.dataset) * ratio / self.num_replicas
        self.num_samples = (math.ceil(target / self.effective_batch_size)
                            * self.effective_batch_size)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class MultiShapeEnlargedSampler(Sampler):
    """Yield accumulation-grouped indices with a shared shape and rotation."""

    def __init__(self, dataset, num_replicas, rank, ratio, batch_size, shapes,
                 use_rot=True, accumulation_steps=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.effective_batch_size = batch_size * accumulation_steps
        self.shapes = [tuple(shape) for shape in shapes]
        self.use_rot = use_rot
        target = len(dataset) * ratio / num_replicas
        self.num_samples = (math.ceil(target / self.effective_batch_size)
                            * self.effective_batch_size)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch * self.num_replicas + self.rank)
        base_indices = torch.randint(
            len(self.dataset), (self.num_samples,), generator=generator).tolist()
        requests = []
        for start in range(0, self.num_samples, self.effective_batch_size):
            shape_idx = torch.randint(len(self.shapes), (1,), generator=generator).item()
            rotation = torch.randint(4, (1,), generator=generator).item() if self.use_rot else 0
            gt_height, gt_width = self.shapes[shape_idx]
            requests.extend((index, gt_height, gt_width, rotation)
                            for index in base_indices[start:start + self.effective_batch_size])
        return iter(requests)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
