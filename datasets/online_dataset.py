"""Online fisheye dataset: render views on the fly (SPEC.md 3.11).

SPEC.md 3.11 says to hold the panoramas in memory.  That does not survive a
Windows DataLoader: ``spawn`` pickles the dataset into every worker, so a
755-panorama train split at 1024x512x3 (about 1.2 GB) is copied once per
worker.  Instead the panoramas are written **once** into a single
``panos.npy`` and opened as a memmap, lazily, inside each worker -- the OS page
cache then holds exactly one copy no matter how many workers there are.

Build the cache before training::

    python -m datasets.online_dataset --root data_root --split train

Per-sample seeds are ``sample_seed(base_seed + epoch, "train_online",
asset_id, view_idx)``, so an epoch is reproducible, successive epochs differ,
and neither depends on worker count or ordering (SPEC.md 3.11, 4.11).  The
epoch comes in with the index, from :class:`datasets.sampler.EpochSampler`,
because a ``persistent_workers`` DataLoader would never see it set on the
dataset itself (SPEC.md 4.12).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.sampling import SamplingConfig, sample_camera, sample_seed
from datasets.fisheye_dataset import IMAGENET_MEAN, IMAGENET_STD, load_norm_stats
from datasets.sampler import split_key
from geometry.render import FisheyeCamera, render_fisheye
from geometry.rotation import rotation_from_ypr

__all__ = [
    "OnlineFisheyeDataset",
    "build_pano_cache",
    "worker_init_fn",
    "PANOS_FILE",
    "INDEX_FILE",
    "main",
]

PANOS_FILE = "panos.npy"
INDEX_FILE = "panos_index.json"
DEFAULT_PANO_SIZE = (1024, 512)  # width, height (SPEC.md 4.4)
SPLIT_TAG = "train_online"


# --- one-off preprocessing --------------------------------------------------


def build_pano_cache(
    root: str | Path,
    asset_ids: Sequence[str],
    cache_dir: str | Path | None = None,
    pano_size: tuple[int, int] = DEFAULT_PANO_SIZE,
    raw_dir: str | Path | None = None,
    manifest: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Downscale the given panoramas into one ``panos.npy`` plus an index.

    Written through a pre-allocated memmap so peak memory is one panorama, not
    the whole split.
    """
    root = Path(root)
    cache_dir = Path(cache_dir) if cache_dir else root / "pano_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(manifest) if manifest else root / "manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))["assets"]
    by_id = {row["asset_id"]: row for row in rows}

    asset_ids = list(asset_ids)
    missing = [a for a in asset_ids if a not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} asset(s) absent from the manifest: {missing[:5]}")

    raw = Path(raw_dir) if raw_dir else root / "raw" / "polyhaven"
    width, height = pano_size
    panos_path = cache_dir / PANOS_FILE

    log(f"caching {len(asset_ids)} panoramas at {width}x{height} -> {panos_path}")
    panos = np.lib.format.open_memmap(
        panos_path, mode="w+", dtype=np.uint8, shape=(len(asset_ids), height, width, 3)
    )
    try:
        for position, asset_id in enumerate(asset_ids, start=0):
            source = raw / by_id[asset_id]["file"]
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"{asset_id}: could not read {source}")
            if (image.shape[1], image.shape[0]) != (width, height):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            panos[position] = image
            if (position + 1) % 100 == 0 or position + 1 == len(asset_ids):
                log(f"  {position + 1}/{len(asset_ids)}")
        panos.flush()
    finally:
        del panos

    (cache_dir / INDEX_FILE).write_text(
        json.dumps(
            {"asset_ids": asset_ids, "pano_size": [width, height], "colour": "BGR"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"wrote {panos_path} and {cache_dir / INDEX_FILE}")
    return cache_dir


def worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` (SPEC.md 4.12).

    Each worker renders on its own core, so OpenCV must not also spawn threads.
    """
    cv2.setNumThreads(0)


# --- dataset ----------------------------------------------------------------


class OnlineFisheyeDataset(Dataset):
    """Render ``views_per_pano`` views of each panorama, per epoch."""

    def __init__(
        self,
        root: str | Path,
        asset_ids: Sequence[str],
        views_per_pano: int,
        sampling_cfg: SamplingConfig,
        base_seed: int,
        transform: Callable[..., np.ndarray] | None = None,
        norm_stats: Mapping[str, Any] | None = None,
        cache_dir: str | Path | None = None,
        supersample: int = 4,
        categories: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.asset_ids = list(asset_ids)
        self.views_per_pano = int(views_per_pano)
        self.sampling_cfg = sampling_cfg
        self.base_seed = int(base_seed)
        self.transform = transform
        self.supersample = int(supersample)
        self.categories = dict(categories or {})

        if self.views_per_pano < 1:
            raise ValueError(f"views_per_pano must be >= 1, got {views_per_pano}")
        if not self.asset_ids:
            raise ValueError("asset_ids is empty")

        self.cache_dir = Path(cache_dir) if cache_dir else self.root / "pano_cache"
        index_path = self.cache_dir / INDEX_FILE
        if not index_path.is_file():
            raise FileNotFoundError(
                f"{index_path} not found; build it with "
                f"`python -m datasets.online_dataset --root {self.root} --split <split>`"
            )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        row_of = {asset_id: i for i, asset_id in enumerate(index["asset_ids"])}
        absent = [a for a in self.asset_ids if a not in row_of]
        if absent:
            raise KeyError(
                f"{len(absent)} asset(s) are not in the pano cache "
                f"(e.g. {absent[:3]}); rebuild it for this split"
            )
        self._rows = [row_of[a] for a in self.asset_ids]
        self.panos_path = self.cache_dir / PANOS_FILE

        self.norm_stats = (
            dict(norm_stats) if norm_stats is not None else load_norm_stats(self.root)
        )
        mean = np.asarray(self.norm_stats["mean"], dtype=np.float64)
        std = np.asarray(self.norm_stats["std"], dtype=np.float64)
        if mean.shape != (2,) or std.shape != (2,):
            raise ValueError("norm_stats mean/std must each have two entries")
        self._target_mean = mean
        self._target_std = np.where(std > 0, std, 1.0)

        # Opened on first use, per process (SPEC.md 4.12).
        self._panos: np.ndarray | None = None

    # --- plumbing ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.asset_ids) * self.views_per_pano

    @property
    def panos(self) -> np.ndarray:
        if self._panos is None:
            self._panos = np.load(self.panos_path, mmap_mode="r")
        return self._panos

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_panos"] = None
        return state

    def locate(self, position: int) -> tuple[str, int]:
        """``position`` to ``(asset_id, view_idx)``.

        Views of one panorama are contiguous, which keeps a worker reading the
        same cache page for a whole run of samples.
        """
        asset_pos, view_idx = divmod(int(position), self.views_per_pano)
        return self.asset_ids[asset_pos], view_idx

    def sample_for(self, position: int, epoch: int = 0):
        """The camera this position renders, without touching any pixels.

        ``__getitem__`` goes through here too, so the reported parameters can
        never drift from the ones actually rendered.
        """
        asset_id, view_idx = self.locate(position)
        return self._sample(asset_id, view_idx, epoch)

    def _sample(self, asset_id: str, view_idx: int, epoch: int):
        seed = sample_seed(self.base_seed + int(epoch), SPLIT_TAG, asset_id, view_idx)
        return sample_camera(np.random.default_rng(seed), self.sampling_cfg)

    # --- rendering ---------------------------------------------------------

    def __getitem__(self, key: Any) -> dict[str, Any]:
        epoch, position = split_key(key)
        asset_pos, view_idx = divmod(position, self.views_per_pano)
        asset_id = self.asset_ids[asset_pos]
        sample = self._sample(asset_id, view_idx, epoch)

        camera = FisheyeCamera(
            f=sample.f,
            xi=sample.xi,
            size=self.sampling_cfg.image_size,
            circular=sample.circular,
        )
        rotation = rotation_from_ypr(sample.yaw_deg, sample.pitch_deg, sample.roll_deg)

        pano = np.ascontiguousarray(self.panos[self._rows[asset_pos]])
        image, _mask = render_fisheye(
            pano, camera, rotation, supersample=self.supersample
        )
        # The cache holds BGR, as cv2 read it; images.npy is RGB (SPEC.md 2.2).
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            aug_seed = (self.base_seed, epoch, asset_pos, view_idx)
            image = self.transform(image, np.random.default_rng(aug_seed))

        # asarray(..., float32) always yields a fresh writable array, so the
        # tensor never aliases the read-only memmap page.
        chw = np.ascontiguousarray(
            np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        tensor = torch.from_numpy(chw)
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std

        raw = np.array([np.log(sample.f), sample.xi], dtype=np.float64)
        target = (raw - self._target_mean) / self._target_std

        return {
            "image": tensor,
            "target": torch.from_numpy(target).float(),
            "f": float(sample.f),
            "xi": float(sample.xi),
            "r_ref": float(sample.r_ref),
            "circular": bool(sample.circular),
            "theta_ref": float(np.radians(sample.fov_deg) / 2.0),
            "asset_id": asset_id,
            "view_idx": view_idx,
            "category": self.categories.get(asset_id, "unknown"),
        }


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the downscaled panorama cache the online dataset reads."
    )
    parser.add_argument("--root", default="data_root")
    parser.add_argument(
        "--split", default="train", help="split from splits.json to cache"
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--width", type=int, default=DEFAULT_PANO_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_PANO_SIZE[1])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    root = Path(args.root)
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    if args.split not in splits:
        raise KeyError(f"unknown split {args.split!r}; have {sorted(splits)}")
    build_pano_cache(
        root=root,
        asset_ids=sorted(splits[args.split]),
        cache_dir=args.cache_dir,
        pano_size=(args.width, args.height),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
