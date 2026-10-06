"""Render the offline fisheye dataset from the Poly Haven panoramas.

SPEC.md 3.9 and 4.11.  Work is dispatched one panorama at a time: a worker
loads that panorama once, renders all of its views and writes them straight
into a pre-allocated memmap at indices it alone owns.

Reproducibility does not depend on the schedule.  A sample's index is fixed by
the sorted asset list, and its seed comes from
``sample_seed(global_seed, split, asset_id, view_idx)``, so rendering with one
worker or eight produces byte-identical output.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from data.sampling import SamplingConfig, sample_camera_with_attempts, sample_seed
from geometry.render import FisheyeCamera, render_fisheye
from geometry.rotation import rotation_from_ypr

__all__ = [
    "LABEL_FIELDS",
    "load_config",
    "texture_score",
    "render_pano",
    "run",
    "main",
]

LABEL_FIELDS = [
    "index", "asset_id", "view_idx", "f", "xi", "fov_deg", "circular",
    "r_ref", "yaw_deg", "pitch_deg", "roll_deg", "category", "texture",
]

SPLIT_CHOICES = ("train", "val", "test", "stress")


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return cfg


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        # e.g. a repository with no commits yet, where rev-parse echoes "HEAD".
        return "unknown"
    return out.stdout.strip() or "unknown"


TEXTURE_ERODE_ITERATIONS = 2
_TEXTURE_KERNEL = np.ones((3, 3), np.uint8)


def texture_score(image: np.ndarray, mask: np.ndarray) -> float:
    """Mean Sobel gradient magnitude inside the valid mask, on the uint8 scale.

    A low score means the frame is mostly flat sky or a blank wall.  It is
    recorded only -- nothing filters on it -- so that Phase 2 can check whether
    featureless views are what the model gets wrong.

    The mask is eroded first.  Invalid pixels are filled with 0, so a Sobel
    window straddling the aperture edge reads that black and reports a steep
    gradient that is an artefact of the fill, not of the scene -- a circular
    fisheye of blank sky would otherwise score like a textured frame.  Two
    3x3 erosions drop both the pixels whose window reaches outside and the
    partially covered ones the INTER_AREA reduction leaves on the rim.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    valid = np.asarray(mask, dtype=bool)
    interior = cv2.erode(
        valid.astype(np.uint8), _TEXTURE_KERNEL, iterations=TEXTURE_ERODE_ITERATIONS
    ).astype(bool)
    if not interior.any():
        return 0.0
    return float(magnitude[interior].mean())


def render_pano(job: Mapping[str, Any]) -> dict[str, Any]:
    """Render every view of one panorama; returns its label rows.

    Runs in a worker process, so it takes plain data and writes the images
    itself.  ``cv2.setNumThreads(0)`` keeps OpenCV from oversubscribing the
    CPU when several workers run at once (SPEC.md 4.11).
    """
    cv2.setNumThreads(0)

    images_path = Path(job["images_path"])
    asset_id = job["asset_id"]
    split = job["split"]
    views = int(job["views"])
    base_index = int(job["base_index"])
    size = int(job["image_size"])
    supersample = int(job["supersample"])
    pano_w, pano_h = job["pano_resize"]
    cfg = SamplingConfig(**job["sampling"])

    pano = cv2.imread(str(job["pano_path"]), cv2.IMREAD_COLOR)
    if pano is None:
        raise RuntimeError(f"{asset_id}: could not read {job['pano_path']}")
    if (pano.shape[1], pano.shape[0]) != (pano_w, pano_h):
        pano = cv2.resize(pano, (pano_w, pano_h), interpolation=cv2.INTER_AREA)
    pano = np.ascontiguousarray(pano)

    images = np.lib.format.open_memmap(images_path, mode="r+")
    rows: list[dict[str, Any]] = []
    attempts_total = 0

    for view_idx in range(views):
        seed = sample_seed(job["seed"], split, asset_id, view_idx)
        rng = np.random.default_rng(seed)
        sample, attempts = sample_camera_with_attempts(rng, cfg)
        attempts_total += attempts

        camera = FisheyeCamera(
            f=sample.f, xi=sample.xi, size=size, circular=sample.circular
        )
        rotation = rotation_from_ypr(
            sample.yaw_deg, sample.pitch_deg, sample.roll_deg
        )
        image, mask = render_fisheye(pano, camera, rotation, supersample=supersample)

        index = base_index + view_idx
        # SPEC.md 2.2 stores RGB; OpenCV works in BGR.
        images[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        row = {"index": index, "asset_id": asset_id, "view_idx": view_idx,
               "category": job["category"],
               "texture": round(texture_score(image, mask), 4)}
        row.update(asdict(sample))
        rows.append({field: row[field] for field in LABEL_FIELDS})

    images.flush()
    del images
    return {"rows": rows, "attempts": attempts_total, "views": views}


def _write_labels(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: int(r["index"])):
            writer.writerow(row)


def _norm_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Standardisation statistics for ``[log f, xi]`` (SPEC.md 4.7)."""
    log_f = np.log(np.array([float(r["f"]) for r in rows], dtype=np.float64))
    xi = np.array([float(r["xi"]) for r in rows], dtype=np.float64)
    stacked = np.stack([log_f, xi], axis=1)
    return {
        "fields": ["log_f", "xi"],
        "mean": stacked.mean(axis=0).tolist(),
        "std": stacked.std(axis=0, ddof=0).tolist(),
    }


def run(
    config: Mapping[str, Any],
    split: str,
    workers: int = 0,
    max_panos: int | None = None,
    out_root: Path | None = None,
    log=print,
) -> dict[str, Any]:
    """Render one split; returns its ``meta.json`` document."""
    if split not in SPLIT_CHOICES:
        raise ValueError(f"split must be one of {SPLIT_CHOICES}, got {split!r}")

    data_root = Path(out_root or config.get("data_root", "data_root"))
    manifest = json.loads(
        Path(config.get("manifest", data_root / "manifest.json")).read_text("utf-8")
    )
    splits = json.loads(
        Path(config.get("splits", data_root / "splits.json")).read_text("utf-8")
    )
    by_id = {row["asset_id"]: row for row in manifest["assets"]}

    asset_ids = sorted(splits[split])
    if max_panos is not None and 0 <= int(max_panos) < len(asset_ids):
        # Draw a random subset rather than the alphabetical head: the first N
        # ids cluster by location and category, which makes a small sample
        # unrepresentative.  The config seed keeps the choice reproducible.
        picker = np.random.default_rng(int(config.get("seed", 0)))
        chosen = picker.choice(len(asset_ids), size=int(max_panos), replace=False)
        asset_ids = sorted(asset_ids[int(i)] for i in chosen)
    views = int(config["views_per_pano"][split])
    if not asset_ids:
        raise ValueError(f"split {split!r} selects no panoramas")

    size = int(config.get("image_size", 224))
    out_dir = data_root / "rendered" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    images_path = out_dir / "images.npy"

    total = len(asset_ids) * views
    log(f"{split}: {len(asset_ids)} panoramas x {views} views = {total} images")

    # Pre-allocate so workers can write their own slice (SPEC.md 4.11).
    images = np.lib.format.open_memmap(
        images_path, mode="w+", dtype=np.uint8, shape=(total, size, size, 3)
    )
    images.flush()
    del images

    raw_dir = Path(config.get("raw_dir", data_root / "raw" / "polyhaven"))
    sampling = SamplingConfig.from_mapping(config)
    jobs = [
        {
            "images_path": str(images_path),
            "pano_path": str(raw_dir / by_id[asset_id]["file"]),
            "asset_id": asset_id,
            "category": by_id[asset_id].get("primary_category", "unknown"),
            "split": split,
            "views": views,
            "base_index": position * views,
            "seed": int(config.get("seed", 0)),
            "image_size": size,
            "supersample": int(config.get("supersample", 4)),
            "pano_resize": list(config.get("pano_resize", [1024, 512])),
            "sampling": asdict(sampling),
        }
        for position, asset_id in enumerate(asset_ids)
    ]

    started = time.time()
    rows: list[dict[str, Any]] = []
    attempts = accepted = 0

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(render_pano, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                rows.extend(result["rows"])
                attempts += result["attempts"]
                accepted += result["views"]
                if done % 25 == 0 or done == len(futures):
                    log(f"  {done}/{len(futures)} panoramas")
    else:
        for done, job in enumerate(jobs, start=1):
            result = render_pano(job)
            rows.extend(result["rows"])
            attempts += result["attempts"]
            accepted += result["views"]
            if done % 25 == 0 or done == len(jobs):
                log(f"  {done}/{len(jobs)} panoramas")

    _write_labels(out_dir / "labels.csv", rows)

    meta: dict[str, Any] = {
        "split": split,
        "config": dict(config),
        "seed": int(config.get("seed", 0)),
        "git_commit": _git_commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "image_size": size,
        "n_panos": len(asset_ids),
        "views_per_pano": views,
        "n_images": total,
        "max_panos": max_panos,
        "rejection_rate": (attempts - accepted) / attempts if attempts else 0.0,
        "sampling_attempts": attempts,
        "elapsed_sec": round(time.time() - started, 2),
    }
    if split == "train":
        # Every split and subset uses the train statistics (SPEC.md 4.7).
        meta["norm_stats"] = _norm_stats(rows)

    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(
        f"wrote {images_path} ({total} images), labels.csv, meta.json; "
        f"rejection rate {meta['rejection_rate']:.3f}, "
        f"{meta['elapsed_sec']:.1f}s"
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the offline fisheye dataset.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    parser.add_argument("--workers", type=int, default=0, help="0 or 1 = serial")
    parser.add_argument(
        "--max-panos", type=int, default=None, help="render only the first N panoramas"
    )
    parser.add_argument("--out-root", default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run(
        config=load_config(Path(args.config)),
        split=args.split,
        workers=args.workers,
        max_panos=args.max_panos,
        out_root=Path(args.out_root) if args.out_root else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
