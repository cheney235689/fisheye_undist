"""Summarise the category structure of a Poly Haven manifest.

SPEC.md 4.10 leaves the decision of which categories to exclude to a human:
this tool only reports, it never filters.  It prints three tables and writes
the same content as Markdown:

(a) how many assets carry each category
(b) how many categories each asset carries
(c) which categories co-occur, among the most common ones

Poly Haven exposes two different category fields, and both are reported
because they answer different questions:

``primary_category``
    First segment of the hierarchical ``category`` path, exactly one per
    asset (12 values).  This is the stratification key for splitting.
``categories``
    A flat list of 4-9 tags per asset (23 values), mixing scene, lighting,
    contrast and time-of-day.  Only this field can co-occur, so table (c) is
    built from it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_MANIFEST = "data_root/manifest.json"
DEFAULT_OUT = "data_root/category_report.md"
DEFAULT_TOP_N = 12

__all__ = [
    "load_assets",
    "category_counts",
    "primary_counts",
    "labels_per_asset",
    "cooccurrence",
    "render_report",
    "main",
]


# --- loading ---------------------------------------------------------------


def load_assets(manifest_path: Path) -> list[dict[str, Any]]:
    """Read the asset rows from a manifest written by ``data.download_polyhaven``."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assets = payload.get("assets") if isinstance(payload, dict) else payload
    if not isinstance(assets, list):
        raise ValueError(f"{manifest_path}: no 'assets' list found")
    return [row for row in assets if isinstance(row, dict)]


# --- statistics ------------------------------------------------------------


def category_counts(assets: Sequence[dict[str, Any]]) -> Counter[str]:
    """How many assets carry each flat ``categories`` tag."""
    counts: Counter[str] = Counter()
    for row in assets:
        counts.update(set(row.get("categories") or []))
    return counts


def primary_counts(assets: Sequence[dict[str, Any]]) -> Counter[str]:
    """How many assets fall in each ``primary_category``."""
    return Counter(row.get("primary_category") or "unknown" for row in assets)


def labels_per_asset(assets: Sequence[dict[str, Any]]) -> Counter[int]:
    """Distribution of the number of flat categories per asset."""
    return Counter(len(set(row.get("categories") or [])) for row in assets)


def cooccurrence(
    assets: Sequence[dict[str, Any]], labels: Sequence[str]
) -> list[list[int]]:
    """Symmetric matrix of how many assets carry both labels.

    The diagonal is the plain count of that label, which makes each row read as
    "of the N assets with this label, how many also have that one".
    """
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for row in assets:
        present = sorted(
            {index[c] for c in set(row.get("categories") or []) if c in index}
        )
        for a_pos, i in enumerate(present):
            matrix[i][i] += 1
            for j in present[a_pos + 1 :]:
                matrix[i][j] += 1
                matrix[j][i] += 1
    return matrix


# --- rendering -------------------------------------------------------------


def _text_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Label columns and the unnamed bar column read left-to-right; counts right.
    left = {i for i, h in enumerate(headers) if i == 0 or h == ""}

    def line(cells: Sequence[str]) -> str:
        parts = [
            cell.ljust(widths[i]) if i in left else cell.rjust(widths[i])
            for i, cell in enumerate(cells)
        ]
        return "  ".join(parts).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    align = ["---"] + ["---:"] * (len(headers) - 1)
    out.append("| " + " | ".join(align) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def _bar(count: int, total: int, width: int = 24) -> str:
    filled = 0 if total <= 0 else round(width * count / total)
    return "#" * filled


def render_report(
    assets: Sequence[dict[str, Any]], top_n: int = DEFAULT_TOP_N, markdown: bool = False
) -> str:
    """Build the whole report; ``markdown`` switches table style."""
    table = _md_table if markdown else _text_table
    total = len(assets)
    heading = (lambda t: f"\n## {t}\n") if markdown else (lambda t: f"\n{t}\n{'=' * len(t)}")

    parts: list[str] = []
    title = "Poly Haven category report"
    parts.append(f"# {title}" if markdown else f"{title}\n{'=' * len(title)}")
    parts.append(f"\n{total} assets in manifest.\n")

    # (a1) primary category -- the stratification key
    parts.append(heading("(a1) Primary category (one per asset, stratification key)"))
    primary = primary_counts(assets)
    rows = [
        [name, str(n), f"{100.0 * n / total:.1f}%", _bar(n, total)]
        for name, n in primary.most_common()
    ]
    parts.append(table(["primary_category", "assets", "share", ""], rows))

    # (a2) flat categories
    parts.append(heading("(a2) Flat categories (4-9 per asset, overlapping)"))
    flat = category_counts(assets)
    rows = [
        [name, str(n), f"{100.0 * n / total:.1f}%", _bar(n, total)]
        for name, n in flat.most_common()
    ]
    parts.append(table(["category", "assets", "share", ""], rows))

    # (b) labels per asset
    parts.append(heading("(b) Number of flat categories per asset"))
    per_asset = labels_per_asset(assets)
    rows = [
        [str(k), str(per_asset[k]), f"{100.0 * per_asset[k] / total:.1f}%",
         _bar(per_asset[k], total)]
        for k in sorted(per_asset)
    ]
    parts.append(table(["labels", "assets", "share", ""], rows))

    # (c) co-occurrence among the most common flat categories
    labels = [name for name, _ in flat.most_common(top_n)]
    parts.append(heading(f"(c) Co-occurrence of the {len(labels)} most common categories"))
    if not labels:
        parts.append("(no categories present)")
        return "\n".join(parts) + "\n"

    matrix = cooccurrence(assets, labels)
    legend_rows = [[str(i), name, str(matrix[i][i])] for i, name in enumerate(labels)]
    parts.append(table(["#", "category", "assets"], legend_rows))
    parts.append("")
    headers = ["#"] + [str(i) for i in range(len(labels))]
    rows = [
        [f"{i} {labels[i]}"] + [str(matrix[i][j]) for j in range(len(labels))]
        for i in range(len(labels))
    ]
    parts.append(table(headers, rows))
    parts.append(
        "\nDiagonal is the total for that category; off-diagonal is how many "
        "assets carry both."
    )
    return "\n".join(parts) + "\n"


# --- CLI -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report Poly Haven category statistics from a manifest."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT, help="Markdown report path")
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"categories in the co-occurrence matrix (default {DEFAULT_TOP_N})",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    assets = load_assets(Path(args.manifest))

    print(render_report(assets, top_n=args.top_n, markdown=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_report(assets, top_n=args.top_n, markdown=True), encoding="utf-8"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
