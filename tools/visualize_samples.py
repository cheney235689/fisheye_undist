"""Visual check of a rendered split (SPEC.md 4.6 and acceptance #14).

Produces three files:

``samples_grid.png``
    An 8x8 grid chosen to span the parameter space rather than the first 64
    rows -- the sample list is sorted by FOV and swept, so one picture covers
    narrow to very wide, low to high ``xi``, circular and full-frame.  Each
    tile is captioned with its parameters so a wrong label is visible.

``param_distributions.png``
    Histograms of ``f``, ``xi``, FOV and ``texture``, the ``(FOV, xi)`` scatter
    where the feasible region of SPEC.md 4.6 should be plainly visible, and
    ``texture`` against FOV.

``low_texture_grid.png``
    The 16 lowest-``texture`` samples, i.e. the most featureless frames in the
    split.  Nothing filters on ``texture``; this is here so a human can judge
    whether those frames are still legitimate training data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # no display in a headless run

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "load_split",
    "sample_grid",
    "low_texture_grid",
    "plot_distributions",
    "main",
]

GRID = 8
LOW_TEXTURE_COUNT = 16
LOW_TEXTURE_COLS = 4

_TRUE = ("True", "true", "1")


def load_split(root: Path, split: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Memmap the images and read labels.csv for one rendered split."""
    out_dir = Path(root) / "rendered" / split
    images = np.load(out_dir / "images.npy", mmap_mode="r")
    with (out_dir / "labels.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(images):
        raise ValueError(f"{split}: {len(rows)} label rows but {len(images)} images")
    return images, rows


def _has_texture(rows: Sequence[dict[str, Any]]) -> bool:
    return bool(rows) and "texture" in rows[0]


def _texture(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.array([float(r["texture"]) for r in rows])


def _spread_indices(rows: Sequence[dict[str, Any]], count: int) -> list[int]:
    """Indices spanning the FOV range, so the grid is not all one look."""
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["fov_deg"]))
    if len(order) <= count:
        return order
    picks = np.linspace(0, len(order) - 1, count).round().astype(int)
    return [order[int(p)] for p in picks]


def _draw_grid(
    images: np.ndarray,
    rows: Sequence[dict[str, Any]],
    picks: Sequence[int],
    out_path: Path,
    cols: int,
    title: str,
    show_texture: bool = False,
) -> Path:
    """Shared contact-sheet renderer."""
    plots = len(picks)
    grid_rows = int(np.ceil(plots / cols)) if plots else 1
    fig, axes = plt.subplots(
        grid_rows, cols, figsize=(cols * 1.9, grid_rows * 2.05), squeeze=False
    )
    flat = axes.ravel()

    for axis, index in zip(flat, picks):
        row = rows[index]
        axis.imshow(images[index])
        shape = "circ" if row["circular"] in _TRUE else "full"
        caption = (
            f"FOV {float(row['fov_deg']):.0f}  xi {float(row['xi']):.2f}\n"
            f"f {float(row['f']):.0f}  {shape}"
        )
        if show_texture and "texture" in row:
            caption += f"\ntex {float(row['texture']):.1f}"
        axis.set_title(caption, fontsize=6.5, pad=2)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in flat[plots:]:
        axis.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def sample_grid(
    images: np.ndarray, rows: Sequence[dict[str, Any]], out_path: Path
) -> Path:
    """Write the captioned 8x8 contact sheet."""
    picks = _spread_indices(rows, GRID * GRID)
    return _draw_grid(
        images, rows, picks, out_path, GRID,
        f"{len(picks)} of {len(rows)} samples, swept across the FOV range",
    )


def low_texture_grid(
    images: np.ndarray,
    rows: Sequence[dict[str, Any]],
    out_path: Path,
    count: int = LOW_TEXTURE_COUNT,
) -> Path | None:
    """Write the grid of the least textured samples, or None if unavailable."""
    if not _has_texture(rows):
        return None
    order = np.argsort(_texture(rows), kind="stable")
    picks = [int(i) for i in order[:count]]
    values = _texture(rows)[picks]
    return _draw_grid(
        images, rows, picks, out_path, LOW_TEXTURE_COLS,
        f"{len(picks)} lowest-texture samples "
        f"(mean Sobel {values.min():.1f} to {values.max():.1f})",
        show_texture=True,
    )


def plot_distributions(
    rows: Sequence[dict[str, Any]], out_path: Path, meta: dict[str, Any] | None = None
) -> Path:
    """Write the parameter histograms and the (FOV, xi) scatter."""
    f = np.array([float(r["f"]) for r in rows])
    xi = np.array([float(r["xi"]) for r in rows])
    fov = np.array([float(r["fov_deg"]) for r in rows])
    circular = np.array([r["circular"] in _TRUE for r in rows], dtype=bool)
    has_texture = _has_texture(rows)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].hist(f, bins=40, color="#4C72B0")
    axes[0, 0].set_xlabel("f (px)")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_title(f"focal length   min {f.min():.0f}  max {f.max():.0f}")

    axes[0, 1].hist(xi, bins=40, color="#55A868")
    axes[0, 1].set_xlabel("xi")
    axes[0, 1].set_title("UCM xi")

    axes[0, 2].hist(fov, bins=40, color="#C44E52")
    axes[0, 2].set_xlabel("FOV (deg, full angle at r_ref)")
    axes[0, 2].set_title("field of view")

    scatter = axes[1, 0]
    for mask, label, colour in (
        (~circular, "full-frame", "#4C72B0"),
        (circular, "circular", "#DD8452"),
    ):
        scatter.scatter(fov[mask], xi[mask], s=6, alpha=0.5, label=label, color=colour)
    scatter.set_xlabel("FOV (deg)")
    scatter.set_ylabel("xi")
    scatter.set_title("feasible region (SPEC 4.6)")
    scatter.legend(fontsize=8, loc="upper left")

    if has_texture:
        texture = _texture(rows)
        axes[1, 1].hist(texture, bins=40, color="#8172B3")
        axes[1, 1].set_xlabel("texture (mean Sobel, uint8 scale)")
        axes[1, 1].set_ylabel("count")
        axes[1, 1].set_title(
            f"texture   min {texture.min():.1f}  median {np.median(texture):.1f}"
            f"  max {texture.max():.1f}"
        )

        axes[1, 2].scatter(fov, texture, s=6, alpha=0.5, color="#937860")
        axes[1, 2].set_xlabel("FOV (deg)")
        axes[1, 2].set_ylabel("texture")
        axes[1, 2].set_title("texture vs field of view")
    else:
        for axis in (axes[1, 1], axes[1, 2]):
            axis.axis("off")
        axes[1, 1].set_title("texture not in labels.csv", fontsize=9)

    rate = (meta or {}).get("rejection_rate")
    subtitle = f"n = {len(rows)}"
    if rate is not None:
        subtitle += f"   rejection rate {rate:.3f}"
    subtitle += f"   circular {100 * circular.mean():.0f}%"
    fig.suptitle(subtitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a sample grid and parameter distributions for a split."
    )
    parser.add_argument("--root", default="data_root")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--out", default=None, help="output directory (default <root>/figures/<split>)"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    root = Path(args.root)
    images, rows = load_split(root, args.split)

    meta_path = root / "rendered" / args.split / "meta.json"
    meta = json.loads(meta_path.read_text("utf-8")) if meta_path.is_file() else {}

    out_dir = Path(args.out) if args.out else root / "figures" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"wrote {sample_grid(images, rows, out_dir / 'samples_grid.png')}")
    print(
        f"wrote {plot_distributions(rows, out_dir / 'param_distributions.png', meta)}"
    )

    low = low_texture_grid(images, rows, out_dir / "low_texture_grid.png")
    if low is None:
        print("skipped low_texture_grid.png: labels.csv has no texture column")
    else:
        print(f"wrote {low}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
