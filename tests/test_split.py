"""Acceptance criterion #10 (SPEC.md 5) and the grouping rules of SPEC.md 4.10.

Beyond the three splits being pairwise disjoint, the things that actually
prevent leakage here are that no *group* spans two splits -- counting only the
members that are not excluded -- and that the stress list is disjoint from all
of them.

Excluded assets leave their group on their own (SPEC.md 4.10.1), so a group
legitimately has members in both stress and train; ``stress_sibling_split``
records where those siblings went.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data.split import (
    DEFAULT_SUBSETS,
    build_groups,
    build_splits,
    collection_of,
    group_stratum,
    stem_of,
)

SPLITS = ("train", "val", "test")
ALL_LISTS = ("train", "val", "test", "stress")


# --- fixtures ---------------------------------------------------------------


def _asset(asset_id: str, primary: str, categories: list[str] | None = None) -> dict:
    return {
        "asset_id": asset_id,
        "primary_category": primary,
        "categories": categories if categories is not None else ["outdoor"],
    }


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    """A synthetic manifest with families, a collection and excluded assets."""
    out: list[dict[str, Any]] = []
    # Ten numbered families per category, so every stratum has enough groups.
    for category in ("Interiors", "Streets & Town", "Forest & Woodland",
                     "Coast & Water"):
        tag = category.split()[0].lower().strip("&")
        for family in range(10):
            for member in range(3):
                out.append(_asset(f"{tag}_scene{family:02d}_{member + 1:02d}", category))
    # A collection that spans two stems and two categories.
    out.append(_asset("alpha_ridge", "Mountains & Hills", ["nature", "collection: trail"]))
    out.append(_asset("beta_valley_dawn", "Mountains & Hills", ["nature", "collection: trail"]))
    out.append(_asset("beta_valley_dusk", "Mountains & Hills", ["nature"]))
    # A family where one member is a Pure Skies variant.
    out.append(_asset("gamma_field", "Fields & Countryside"))
    out.append(_asset("gamma_field_puresky", "Pure Skies"))
    # A standalone excluded asset.
    out.append(_asset("lonely_sky", "Pure Skies"))
    return out


@pytest.fixture
def document(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return build_splits(rows, seed=0)


# --- grouping helpers -------------------------------------------------------


@pytest.mark.parametrize(
    "asset_id,expected",
    [
        ("qwantani_afternoon_puresky", "qwantani"),
        ("kloofendal_38d_partly_cloudy", "kloofendal"),
        ("wooden_studio_07", "wooden_studio"),
        ("aarfontein_dawn_2", "aarfontein"),
        ("syferfontein_18d_clear", "syferfontein"),
        ("moon_lab", "moon_lab"),
        ("approaching_storm", "approaching_storm"),
    ],
)
def test_stem_of_strips_the_expected_tails(asset_id: str, expected: str) -> None:
    assert stem_of(asset_id) == expected


def test_stem_never_empties_the_id() -> None:
    for asset_id in ("night", "puresky", "01", "2k"):
        assert stem_of(asset_id) == asset_id


def test_collection_of_reads_the_tag() -> None:
    assert collection_of({"categories": ["nature", "collection: trail"]}) == "collection: trail"
    assert collection_of({"categories": ["nature"]}) is None
    assert collection_of({}) is None


def test_groups_merge_collection_and_stem(rows: list[dict[str, Any]]) -> None:
    """The union of both relations, not a precedence between them.

    ``beta_valley_dusk`` has no collection tag but shares a stem with
    ``beta_valley_dawn``, which does; all three must land together.
    """
    groups = build_groups(rows)
    home = {asset: key for key, members in groups.items() for asset in members}
    assert home["alpha_ridge"] == home["beta_valley_dawn"] == home["beta_valley_dusk"]


def test_group_stratum_takes_the_mode(rows: list[dict[str, Any]]) -> None:
    by_id = {r["asset_id"]: r for r in rows}
    assert group_stratum(["gamma_field", "gamma_field_puresky"], by_id) in (
        "Fields & Countryside",
        "Pure Skies",
    )
    assert group_stratum(["gamma_field"], by_id) == "Fields & Countryside"


# --- #10 no leakage ---------------------------------------------------------


def test_splits_are_pairwise_disjoint(document: dict[str, Any]) -> None:
    """#10: the three splits share no asset_id."""
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not set(document[a]) & set(document[b]), f"{a} & {b}"


def test_stress_is_disjoint_from_every_split(document: dict[str, Any]) -> None:
    stress = set(document["stress"])
    assert stress
    for name in SPLITS:
        assert not stress & set(document[name]), name


def test_every_asset_lands_exactly_once(
    document: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    seen = [asset for name in ALL_LISTS for asset in document[name]]
    assert len(seen) == len(set(seen)), "an asset appears in two lists"
    assert set(seen) == {r["asset_id"] for r in rows}


def test_non_excluded_group_members_never_span_splits(
    document: dict[str, Any]
) -> None:
    """The real leakage guard, applied to the members that stay in the core.

    An excluded asset is expected to sit in stress while its siblings are in
    train; what must never happen is two *non-excluded* members of one group
    landing in different splits.
    """
    where = {asset: name for name in ALL_LISTS for asset in document[name]}
    stress = set(document["stress"])
    for key, members in document["groups"].items():
        kept = [asset for asset in members if asset not in stress]
        homes = {where[asset] for asset in kept}
        assert len(homes) <= 1, f"group {key} spans {sorted(homes)}"


def test_only_the_matching_asset_is_excluded(document: dict[str, Any]) -> None:
    """SPEC.md 4.10.1: exclusion is per asset; siblings stay in the core."""
    stress = set(document["stress"])
    assert {"gamma_field_puresky", "lonely_sky"} <= stress
    assert "gamma_field" not in stress
    assert "gamma_field" in set(document["train"]) | set(document["val"]) | set(
        document["test"]
    )


def test_every_stress_asset_matches_an_excluded_category(
    document: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    by_id = {r["asset_id"]: r for r in rows}
    excluded = set(document["exclude_categories"])
    assert excluded
    for asset_id in document["stress"]:
        assert by_id[asset_id]["primary_category"] in excluded, asset_id


def test_stress_sibling_split_is_recorded(document: dict[str, Any]) -> None:
    """One entry per stress asset, naming where its siblings went."""
    sibling = document["stress_sibling_split"]
    assert set(sibling) == set(document["stress"])

    where = {asset: name for name in SPLITS for asset in document[name]}
    stress = set(document["stress"])
    group_of = {
        asset: key for key, members in document["groups"].items() for asset in members
    }

    for asset_id, recorded in sibling.items():
        kept = [
            other
            for other in document["groups"][group_of[asset_id]]
            if other not in stress
        ]
        if kept:
            assert recorded in SPLITS
            assert {where[other] for other in kept} == {recorded}
        else:
            assert recorded is None

    # The synthetic manifest covers both branches.
    assert sibling["gamma_field_puresky"] in SPLITS
    assert sibling["lonely_sky"] is None


def test_nothing_excluded_means_no_stress(rows: list[dict[str, Any]]) -> None:
    document = build_splits(rows, seed=0, exclude_categories=[])
    assert document["stress"] == []
    total = sum(len(document[name]) for name in SPLITS)
    assert total == len(rows)


# --- stratification and subsets ---------------------------------------------


def test_every_large_stratum_appears_in_all_three_splits(
    document: dict[str, Any]
) -> None:
    """SPEC.md 4.10.3: at least one group per split when a stratum has >= 3."""
    strata = document["group_stratum"]
    counts: dict[str, int] = {}
    for key, stratum in strata.items():
        counts[stratum] = counts.get(stratum, 0) + 1

    for name in SPLITS:
        present = {strata[key] for key in document["group_split"][name]}
        for stratum, n in counts.items():
            if n >= 3:
                assert stratum in present, f"{stratum} missing from {name}"


def test_subsets_are_nested(document: dict[str, Any]) -> None:
    subsets = document["train_subsets"]
    p10, p30, p100 = (set(subsets[k]) for k in ("p10", "p30", "p100"))
    assert p10 <= p30 <= p100
    assert p100 == set(document["train"])
    assert 0 < len(p10) < len(p30) < len(p100)


def test_subsets_keep_every_stratum(document: dict[str, Any]) -> None:
    strata = document["group_stratum"]
    train_strata = {strata[k] for k in document["group_split"]["train"]}
    groups = document["core_groups"]
    for name in DEFAULT_SUBSETS:
        members = set(document["train_subsets"][name])
        present = {
            strata[key] for key, assets in groups.items()
            if key in strata and set(assets) <= members
        }
        assert present == train_strata, f"{name} dropped a stratum"


def test_subsets_respect_group_boundaries(document: dict[str, Any]) -> None:
    """A subset takes whole groups, never part of one."""
    groups = document["core_groups"]
    for name in DEFAULT_SUBSETS:
        members = set(document["train_subsets"][name])
        for key, assets in groups.items():
            overlap = members & set(assets)
            assert overlap in (set(), set(assets)), f"{name} splits group {key}"


# --- determinism ------------------------------------------------------------


def test_same_seed_gives_the_same_split(rows: list[dict[str, Any]]) -> None:
    a = build_splits(rows, seed=0)
    b = build_splits(rows, seed=0)
    assert a == b


def test_a_different_seed_moves_groups(rows: list[dict[str, Any]]) -> None:
    a = build_splits(rows, seed=0)
    b = build_splits(rows, seed=1)
    assert a["train"] != b["train"] or a["val"] != b["val"]
    # The grouping itself must not depend on the seed.
    assert a["groups"] == b["groups"]


def test_manifest_order_does_not_matter(rows: list[dict[str, Any]]) -> None:
    shuffled = list(reversed(rows))
    assert build_splits(rows, seed=0) == build_splits(shuffled, seed=0)


def test_ratios_are_approximately_respected(rows: list[dict[str, Any]]) -> None:
    document = build_splits(rows, seed=0)
    counts = [len(document[name]) for name in SPLITS]
    total = sum(counts)
    train_share = counts[0] / total
    assert 0.70 < train_share < 0.90, train_share


# --- the real split, when it has been generated ------------------------------

REAL = Path("data_root/splits.json")


@pytest.mark.skipif(not REAL.is_file(), reason="data_root/splits.json not generated")
def test_real_split_has_no_leakage() -> None:
    document = json.loads(REAL.read_text(encoding="utf-8"))
    where = {asset: name for name in ALL_LISTS for asset in document[name]}

    for i, a in enumerate(ALL_LISTS):
        for b in ALL_LISTS[i + 1:]:
            assert not set(document[a]) & set(document[b]), f"{a} & {b}"

    stress = set(document["stress"])
    for key, members in document["groups"].items():
        kept = [asset for asset in members if asset not in stress]
        homes = {where[asset] for asset in kept}
        assert len(homes) <= 1, f"group {key} spans {sorted(homes)}"

    sibling = document["stress_sibling_split"]
    assert set(sibling) == stress
    assert all(v in (None, *SPLITS) for v in sibling.values())

    subsets = document["train_subsets"]
    assert set(subsets["p10"]) <= set(subsets["p30"]) <= set(subsets["p100"])
    assert set(subsets["p100"]) == set(document["train"])
