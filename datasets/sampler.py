"""Epoch-carrying sampler (SPEC.md 4.12).

A ``DataLoader`` started with ``persistent_workers=True`` keeps its worker
processes alive across epochs, and each worker holds its own pickled copy of
the dataset.  Mutating the dataset in the training loop -- the usual
``dataset.set_epoch(e)`` -- therefore changes only the parent's copy, and the
workers keep augmenting and sampling exactly as they did in epoch 0.  The bug
is silent: training runs, it just never sees new augmentations.

The sampler, by contrast, runs in the parent process and its output is what
gets shipped to the workers.  So the epoch travels with the index: this
sampler yields ``(epoch, index)`` pairs and the datasets derive their seeds
from the pair.  Nothing in a worker has to be told anything.
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import Sampler

__all__ = ["EpochSampler", "split_key"]


def split_key(key: object) -> tuple[int, int]:
    """Unpack a dataset key into ``(epoch, index)``.

    A bare ``int`` means epoch 0, so a dataset stays usable with a plain
    ``dataset[i]``, a default ``DataLoader``, or any other sampler.
    """
    if isinstance(key, (tuple, list)):
        if len(key) != 2:
            raise ValueError(f"expected a (epoch, index) pair, got {key!r}")
        return int(key[0]), int(key[1])
    if isinstance(key, torch.Tensor):
        return 0, int(key.item())
    return 0, int(key)


class EpochSampler(Sampler):
    """Yield ``(epoch, index)`` pairs, reshuffling per epoch from a fixed seed.

    ``set_epoch`` must be called by the training loop before each epoch, as
    with ``DistributedSampler``; unlike a dataset's, this call is effective
    under ``persistent_workers`` because the sampler is consumed in the parent.
    """

    def __init__(self, num_samples: int, shuffle: bool = True, seed: int = 0) -> None:
        self.num_samples = int(num_samples)
        if self.num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[int, int]]:
        if self.shuffle:
            generator = torch.Generator()
            # Mixing rather than adding keeps seed 1/epoch 0 and seed 0/epoch 1
            # from colliding.
            generator.manual_seed(self.seed * 1_000_003 + self.epoch)
            order = torch.randperm(self.num_samples, generator=generator).tolist()
        else:
            order = range(self.num_samples)
        epoch = self.epoch
        for index in order:
            yield (epoch, int(index))
