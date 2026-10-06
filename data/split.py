"""Group-aware, stratified train/val/test split of the Poly Haven manifest.

SPEC.md 4.10.  Poly Haven ships many variants of the same capture -- the same
location at several times of day, ``_puresky`` versions, numbered series -- so
splitting per asset would put the same scene in both train and test.  Assets
are therefore grouped first and whole groups are assigned to a split.

Two relations put assets in the same group:

``collection``
    A flat category of the form ``"collection: xxx"``.
``stem``
    The ``asset_id`` with trailing numeric, resolution, sun-elevation and
    time/weather tokens stripped (see :data:`STRIP_TOKENS`).

Both are equivalence relations, so the real grouping is their union -- the
connected components of "same collection or same stem".  Taking either as a
precedence rule instead would split genuine families: ``kloofendal_43d_clear``
carries ``collection: verdant_trail`` while ``kloofendal_43d_clear_puresky``
does not, yet they are the same capture.

Assets whose ``primary_category`` is excluded go to the ``stress`` list on
their own; their group siblings stay in train/val/test as usual.  That overlap
is deliberate -- see SPEC.md 4.10.1 -- and ``stress_sibling_split`` records,
for each stress asset, which split its siblings landed in so Phase 2 can
measure what the shared scenery is worth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DEFAULT_MANIFEST = "data_root/manifest.json"
DEFAULT_OUT = "data_root/splits.json"
DEFAULT_SEED = 0
DEFAULT_SPLIT_RATIO = (0.8, 0.1, 0.1)
DEFAULT_SUBSETS: dict[str, float] = {"p10": 0.10, "p30": 0.30, "p100": 1.0}
DEFAULT_EXCLUDE_CATEGORIES = ("Pure Skies",)

SPLIT_NAMES = ("train", "val", "test")

# Trailing tokens stripped to find an asset's stem.  Derived by inspecting all
# 997 ids: every entry here actually occurs as a trailing segment, and the
# scene-name words that merely look like conditions (e.g. "storm" in
# "approaching_storm") are deliberately absent.
STRIP_TOKENS = frozenset(
    {
        # time of day
        "dawn", "dusk", "sunrise", "sunset", "morning", "afternoon",
        "evening", "noon", "midday", "night", "day",
        # sky / weather
        "puresky", "sky", "clear", "cloudy", "partly", "overcast",
        "sunny", "foggy", "misty", "moonlit",
        # season
        "spring", "summer", "autumn", "snow",
    }
)

# Pure number (_01, _2), resolution (_2k), or sun elevation (_28d).
_NUMERIC_TAIL = re.compile(r"\d+[kd]?", re.IGNORECASE)

__all__ = [
    "collection_of",
    "stem_of",
    "build_groups",
    "group_stratum",
    "assign_groups",
    "nested_subsets",
    "build_splits",
    "main",
]


# --- grouping ---------------------------------------------------------------


def collection_of(row: Mapping[str, Any]) -> str | None:
    """The ``collection: xxx`` tag of a manifest row, if it has one."""
    for category in row.get("categories") or []:
        text = str(category).strip()
        if text.lower().startswith("collection:"):
            return text
    return None


def stem_of(asset_id: str) -> str:
    """Strip trailing number / resolution / sun-angle / condition tokens.

    ``qwantani_afternoon_puresky`` -> ``qwantani``,
    ``kloofendal_38d_partly_cloudy`` -> ``kloofendal``.
    At least one segment is always kept.
    """
    parts = str(asset_id).split("_")
    while len(parts) > 1:
        last = parts[-1].lower()
        if _NUMERIC_TAIL.fullmatch(last) or last in STRIP_TOKENS:
            parts.pop()
            continue
        break
    return "_".join(parts)


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Keep the lexicographically smaller root so grouping is stable.
            low, high = sorted((root_a, root_b))
            self._parent[high] = low


def build_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Connected components of "same collection or same stem".

    Returns ``{group_key: [asset_id, ...]}`` with both the keys and each member
    list sorted, so the grouping is independent of manifest order.
    """
    union = _UnionFind()
    for row in rows:
        asset_id = str(row["asset_id"])
        union.union(f"asset:{asset_id}", f"stem:{stem_of(asset_id)}")
        collection = collection_of(row)
        if collection:
            union.union(f"asset:{asset_id}", f"collection:{collection}")

    members: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        asset_id = str(row["asset_id"])
        members[union.find(f"asset:{asset_id}")].append(asset_id)

    # Name each group after its alphabetically first member, which is stable
    # and readable regardless of which tag won the union.
    named: dict[str, list[str]] = {}
    for ids in members.values():
        named[sorted(ids)[0]] = sorted(ids)
    return dict(sorted(named.items()))


def group_stratum(
    group: Sequence[str], by_id: Mapping[str, Mapping[str, Any]]
) -> str:
    """Stratum of a group: the modal ``primary_category``, ties alphabetical."""
    counts = Counter(
        str(by_id[asset_id].get("primary_category") or "unknown")
        for asset_id in group
    )
    best = max(counts.values())
    return sorted(name for name, n in counts.items() if n == best)[0]


# --- assignment -------------------------------------------------------------


def _largest_remainder(total: int, ratios: Sequence[float]) -> list[int]:
    """Split ``total`` into len(ratios) integers summing exactly to ``total``."""
    exact = [total * r for r in ratios]
    counts = [int(np.floor(value)) for value in exact]
    remainder = total - sum(counts)
    order = np.argsort([-(value - np.floor(value)) for value in exact])
    for position in range(remainder):
        counts[int(order[position % len(counts)])] += 1
    return counts


def assign_groups(
    groups: Mapping[str, Sequence[str]],
    strata: Mapping[str, str],
    rng: np.random.Generator,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIO,
    min_groups_for_floor: int = 3,
) -> dict[str, list[str]]:
    """Assign whole groups to train/val/test, stratified by ``strata``.

    Every stratum with at least ``min_groups_for_floor`` groups is guaranteed
    one group in each split; smaller strata just follow the ratio.
    """
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"expected {len(SPLIT_NAMES)} ratios, got {ratios}")

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for key in sorted(groups):
        by_stratum[strata[key]].append(key)

    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for stratum in sorted(by_stratum):
        keys = list(by_stratum[stratum])
        rng.shuffle(keys)  # keys arrive sorted, so this is seed-reproducible
        counts = _largest_remainder(len(keys), ratios)

        if len(keys) >= min_groups_for_floor:
            # Lift every split to at least one group, paying from the largest.
            for index in range(len(counts)):
                if counts[index] == 0:
                    donor = int(np.argmax(counts))
                    counts[donor] -= 1
                    counts[index] += 1

        start = 0
        for name, count in zip(SPLIT_NAMES, counts):
            assigned[name].extend(keys[start : start + count])
            start += count

    return {name: sorted(keys) for name, keys in assigned.items()}


def nested_subsets(
    train_keys: Sequence[str],
    strata: Mapping[str, str],
    rng: np.random.Generator,
    fractions: Mapping[str, float] = DEFAULT_SUBSETS,
) -> dict[str, list[str]]:
    """Nested, stratified subsets of the train groups: p10 subset of p30 ...

    Each stratum is shuffled once and the subsets take growing prefixes of that
    one order, which is what makes them nested by construction.
    """
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for key in sorted(train_keys):
        by_stratum[strata[key]].append(key)
    for keys in by_stratum.values():
        rng.shuffle(keys)

    subsets: dict[str, list[str]] = {}
    for name in sorted(fractions, key=lambda n: fractions[n]):
        fraction = float(fractions[name])
        chosen: list[str] = []
        for stratum in sorted(by_stratum):
            keys = by_stratum[stratum]
            # At least one group per stratum, so no stratum vanishes at p10.
            take = len(keys) if fraction >= 1.0 else max(1, int(round(len(keys) * fraction)))
            chosen.extend(keys[:take])
        subsets[name] = sorted(chosen)
    return subsets


# --- top level --------------------------------------------------------------


def build_splits(
    rows: Sequence[Mapping[str, Any]],
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIO,
    subsets: Mapping[str, float] = DEFAULT_SUBSETS,
    exclude_categories: Iterable[str] = DEFAULT_EXCLUDE_CATEGORIES,
) -> dict[str, Any]:
    """Produce the ``splits.json`` document."""
    by_id = {str(row["asset_id"]): row for row in rows}
    excluded = {str(name) for name in exclude_categories}

    # Grouping is computed over every asset, so a family is recognised whole.
    groups = build_groups(rows)

    # Only the matching asset itself goes to stress (SPEC.md 4.10.1); its
    # siblings stay in the core splits.
    stress = sorted(
        asset_id
        for asset_id, row in by_id.items()
        if str(row.get("primary_category") or "unknown") in excluded
    )
    stress_set = set(stress)

    # A group contributes whatever members survive the exclusion.  Groups left
    # with nothing simply drop out of the core splits.
    core_groups = {
        key: [asset_id for asset_id in members if asset_id not in stress_set]
        for key, members in groups.items()
    }
    core_groups = {key: members for key, members in core_groups.items() if members}
    strata = {key: group_stratum(members, by_id) for key, members in core_groups.items()}

    rng = np.random.default_rng(seed)
    assigned = assign_groups(core_groups, strata, rng, ratios)
    subset_keys = nested_subsets(assigned["train"], strata, rng, subsets)

    # Where did each stress asset's siblings end up?  None means the group had
    # no surviving member (every one of them was excluded).
    split_of_group = {
        key: name for name in SPLIT_NAMES for key in assigned[name]
    }
    group_of_asset = {
        asset_id: key for key, members in groups.items() for asset_id in members
    }
    stress_sibling_split = {
        asset_id: split_of_group.get(group_of_asset[asset_id]) for asset_id in stress
    }

    def expand(keys: Iterable[str]) -> list[str]:
        return sorted(asset_id for key in keys for asset_id in core_groups[key])

    document: dict[str, Any] = {
        "seed": int(seed),
        "split_ratio": list(ratios),
        "exclude_categories": sorted(excluded),
        "n_groups": len(groups),
        "n_groups_core": len(core_groups),
        "n_groups_fully_excluded": len(groups) - len(core_groups),
        "train": expand(assigned["train"]),
        "val": expand(assigned["val"]),
        "test": expand(assigned["test"]),
        "stress": stress,
        "stress_sibling_split": stress_sibling_split,
        "train_subsets": {
            name: expand(keys) for name, keys in sorted(subset_keys.items())
        },
        "groups": {key: list(members) for key, members in groups.items()},
        "group_split": {name: list(keys) for name, keys in sorted(assigned.items())},
        "core_groups": {key: list(members) for key, members in core_groups.items()},
        "group_stratum": dict(sorted(strata.items())),
    }
    return document


def load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = payload.get("assets") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{manifest_path}: no 'assets' list found")
    return [row for row in rows if isinstance(row, dict) and "asset_id" in row]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Group-aware stratified split of a Poly Haven manifest."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--exclude-categories",
        nargs="*",
        default=list(DEFAULT_EXCLUDE_CATEGORIES),
        help="primary_category values routed to the stress list",
    )
    parser.add_argument(
        "--split-ratio",
        nargs=3,
        type=float,
        default=list(DEFAULT_SPLIT_RATIO),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    rows = load_rows(Path(args.manifest))
    document = build_splits(
        rows,
        seed=args.seed,
        ratios=tuple(args.split_ratio),
        exclude_categories=args.exclude_categories,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"{len(rows)} assets -> {document['n_groups']} groups; "
        f"train {len(document['train'])}, val {len(document['val'])}, "
        f"test {len(document['test'])}, stress {len(document['stress'])}"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
