import math
import torch
import numpy as np
import torch.nn as nn
def expand_prediction(arr):
    arr_reshaped = arr.reshape(-1, 1)
    return np.clip(np.concatenate((1.0 - arr_reshaped, arr_reshaped), axis=1), 0.0, 1.0)

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

              
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count        

class FilterModule(nn.Module):
    def __init__(
        self, size, band_start, band_end, norm=False
    ):
        super(FilterModule, self).__init__()

        self.base = nn.Parameter(
            torch.tensor(generate_filter(band_start, band_end, size)),
            requires_grad=False,
        )
        self.learnable = nn.Parameter(
                torch.randn(size, size), requires_grad=True
            )
        self.learnable.data.normal_(0.0, 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(
                torch.sum(
                    torch.tensor(generate_filter(band_start, band_end, size))
                ),
                requires_grad=False,
            )

    def forward(self, x):
        filt = self.base + norm_sigma(self.learnable)

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class Filter():
    def __init__(
        self, size, band_start, band_end, norm=False):
        self.base = torch.nn.Parameter(
            torch.tensor(generate_filter(band_start, band_end, size)),
            requires_grad=False,
        )
        self.learnable = torch.nn.Parameter(
                torch.randn(size, size), requires_grad=False
            )
        self.learnable.data.normal_(0.0, 0.1)

        self.norm = norm
        if norm:
            self.ft_num = torch.nn.Parameter(
                torch.sum(
                    torch.tensor(generate_filter(band_start, band_end, size))
                ),
                requires_grad=False,
            )

    def forward(self, x):
        filt = self.base + norm_sigma(self.learnable)

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [
        [0.0 if i + j > end or i + j <= start else 1.0 for j in range(size)]
        for i in range(size)
    ]

def norm_sigma(x):
    return 2.0 * torch.sigmoid(x) - 1.0