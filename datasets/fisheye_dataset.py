"""Offline fisheye dataset backed by the rendered memmap (SPEC.md 3.10).

``__getitem__`` takes either a plain index or an ``(epoch, index)`` pair from
:class:`datasets.sampler.EpochSampler`.  The epoch arrives with the index
rather than being set on the dataset, because a ``persistent_workers``
DataLoader never re-pickles the dataset and so would never see a
``set_epoch`` (SPEC.md 4.12).

The images live in one ``images.npy`` per split and are read through a memmap,
which is opened **lazily, inside the worker**.  Windows DataLoaders start
workers with ``spawn``: anything held on the dataset at construction is pickled
into every worker, and a live memmap either fails to pickle or is rebuilt per
worker with no shared page cache.  Opening on first ``__getitem__`` keeps one
handle per process and lets the OS share the pages (SPEC.md 4.12).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.sampler import split_key

__all__ = ["FisheyeDataset", "IMAGENET_MEAN", "IMAGENET_STD", "load_norm_stats"]

# torchvision's standard ImageNet statistics, in RGB order; images.npy is RGB
# (SPEC.md 2.2), so no channel swap is needed.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_TRUE = ("True", "true", "1")


def load_norm_stats(root: Path | str) -> dict[str, Any]:
    """Standardisation statistics from the **train** split (SPEC.md 4.7).

    val, test and stress must all use these; computing their own would make
    their targets incomparable with what the model was trained on.
    """
    meta_path = Path(root) / "rendered" / "train" / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{meta_path} not found; render the train split before any other split"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats = meta.get("norm_stats")
    if not stats:
        raise ValueError(f"{meta_path} has no norm_stats")
    return stats


def _read_labels(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class FisheyeDataset(Dataset):
    """One rendered split, optionally restricted to a train subset.

    ``__getitem__`` returns the standardised target plus the raw parameters, so
    evaluation can compute reprojection error without re-reading labels.csv,
    and ``texture`` so error analysis can ask whether the failures are the
    featureless frames.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        subset: str | None = None,
        transform: Callable[..., np.ndarray] | None = None,
        norm_stats: Mapping[str, Any] | None = None,
        aug_seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.subset = subset
        self.transform = transform
        self.aug_seed = int(aug_seed)

        split_dir = self.root / "rendered" / self.split
        self.images_path = split_dir / "images.npy"
        if not self.images_path.is_file():
            raise FileNotFoundError(f"{self.images_path} not found; render it first")

        rows = _read_labels(split_dir / "labels.csv")
        self.indices = self._select(rows, subset)
        self.rows = [rows[i] for i in self.indices]

        self.norm_stats = (
            dict(norm_stats) if norm_stats is not None else load_norm_stats(self.root)
        )
        mean = np.asarray(self.norm_stats["mean"], dtype=np.float64)
        std = np.asarray(self.norm_stats["std"], dtype=np.float64)
        if mean.shape != (2,) or std.shape != (2,):
            raise ValueError("norm_stats mean/std must each have two entries")
        self._target_mean = mean
        self._target_std = np.where(std > 0, std, 1.0)

        # Opened on first use, in whichever process that happens to be.
        self._images: np.ndarray | None = None

    # --- construction helpers ---------------------------------------------

    def _select(self, rows: Sequence[Mapping[str, str]], subset: str | None) -> list[int]:
        if subset is None:
            return list(range(len(rows)))
        splits_path = self.root / "splits.json"
        if not splits_path.is_file():
            raise FileNotFoundError(f"{splits_path} not found; subset={subset!r} needs it")
        document = json.loads(splits_path.read_text(encoding="utf-8"))
        subsets = document.get("train_subsets", {})
        if subset not in subsets:
            raise KeyError(f"unknown subset {subset!r}; have {sorted(subsets)}")
        allowed = set(subsets[subset])
        keep = [i for i, row in enumerate(rows) if row["asset_id"] in allowed]
        if not keep:
            raise ValueError(f"subset {subset!r} selects no row of split {self.split!r}")
        return keep

    # --- dataset protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def images(self) -> np.ndarray:
        """The memmap, opened on first touch in this process (SPEC.md 4.12)."""
        if self._images is None:
            self._images = np.load(self.images_path, mmap_mode="r")
        return self._images

    def __getstate__(self) -> dict[str, Any]:
        # Never ship an open memmap to a worker.
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def _augment(self, image: np.ndarray, position: int, epoch: int) -> np.ndarray:
        if self.transform is None:
            return image
        # Deterministic per (epoch, sample): reproducible, and independent of
        # worker count or iteration order.
        seed = (self.aug_seed, int(epoch), int(self.rows[position]["index"]))
        return self.transform(image, np.random.default_rng(seed))

    def __getitem__(self, key: Any) -> dict[str, Any]:
        epoch, position = split_key(key)
        row = self.rows[position]
        source = self.indices[position]

        image = np.asarray(self.images[source])
        image = self._augment(image, position, epoch)

        # asarray(..., float32) always yields a fresh writable array, so the
        # tensor never aliases the read-only memmap page.
        chw = np.ascontiguousarray(
            np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        tensor = torch.from_numpy(chw)
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std

        f = float(row["f"])
        xi = float(row["xi"])
        raw = np.array([np.log(f), xi], dtype=np.float64)
        target = (raw - self._target_mean) / self._target_std

        return {
            "image": tensor,
            "target": torch.from_numpy(target).float(),
            "f": f,
            "xi": xi,
            "r_ref": float(row["r_ref"]),
            "circular": row["circular"] in _TRUE,
            "theta_ref": float(np.radians(float(row["fov_deg"])) / 2.0),
            "texture": float(row["texture"]),
            "asset_id": row["asset_id"],
            "category": row["category"],
        }
