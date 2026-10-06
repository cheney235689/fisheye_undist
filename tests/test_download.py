"""Tests for ``data.download_polyhaven``.

No test here touches the network: every run goes through a fake session, and
an autouse fixture additionally poisons ``requests.sessions.Session.request``
so an accidental real call fails loudly instead of silently reaching out.

The fixture payloads are trimmed copies of what the live API actually
returned when SPEC.md 3.7's "query /files/{id} first" step was carried out.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import requests

from data import download_polyhaven as dl


# --- no-network guard -------------------------------------------------------


@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("test attempted a real network request")

    monkeypatch.setattr(requests.sessions.Session, "request", explode)


# --- fixtures mirroring the live API ---------------------------------------

ASSETS_PAYLOAD: dict[str, Any] = {
    "beta_asset": {
        "name": "Beta Asset",
        "categories": ["outdoor", "nature", "natural light", "clear"],
        "category": "Mountains & Hills/Alpine/Ridgelines",
        "attributes": {"environment": "outdoor", "contrast": "high"},
        "authors": {"Someone": "All"},
        "max_resolution": [16384, 8192],
    },
    "alpha_asset": {
        "name": "Alpha Asset",
        "categories": ["indoor", "artificial light", "studio", "high contrast"],
        "category": "Studio/Seamless/White",
        "attributes": {"environment": "indoor", "contrast": "high"},
        "authors": {"Another": "All"},
        "max_resolution": [16384, 8192],
    },
    "gamma_asset": {
        "name": "Gamma Asset",
        "categories": ["outdoor", "urban", "natural light", "overcast"],
        "category": "Streets & Town/European/Alleys",
        "attributes": {"environment": "outdoor"},
        "authors": {"Third": "All"},
        "max_resolution": [16384, 8192],
    },
}


def _files_payload(asset_id: str) -> dict[str, Any]:
    base = "https://dl.polyhaven.org/file/ph-assets/HDRIs"
    return {
        "hdri": {
            res: {
                "exr": {"url": f"{base}/exr/{res}/{asset_id}_{res}.exr", "md5": "e" * 32,
                        "size": 21448332},
                "hdr": {"url": f"{base}/hdr/{res}/{asset_id}_{res}.hdr",
                        "md5": None, "size": 6156727},
            }
            for res in ("1k", "2k", "4k", "8k", "16k")
        },
        "tonemapped": {
            "url": f"{base}/extra/Tonemapped%20JPG/{asset_id}.jpg",
            "md5": "t" * 32,
            "size": 23626563,
        },
        "colorchart": {"url": f"{base}/extra/Color%20Charts/{asset_id}.zip",
                       "md5": "c" * 32, "size": 568881669},
    }


HDR_H, HDR_W = 64, 128


def _hdr_bytes(asset_id: str, tmp_path: Path) -> bytes:
    """A real Radiance .hdr payload so cv2 can decode it like the live file.

    The content is smooth and photo-like rather than white noise: JPEG is
    designed for natural images, and compressing noise at quality 95 costs tens
    of grey levels, which would say nothing about the tonemap.  A per-asset
    phase keeps the three fixtures distinct so their MD5s differ.
    """
    phase = (abs(hash(asset_id)) % 360) * np.pi / 180.0
    yy, xx = np.mgrid[0:HDR_H, 0:HDR_W].astype(np.float32)
    u = xx / HDR_W * 2.0 * np.pi
    v = yy / HDR_H * np.pi
    base = 1.5 + 1.2 * np.sin(u + phase) * np.cos(v)
    radiance = np.stack(
        [base, base * (1.0 + 0.3 * np.cos(v)), base * (1.0 + 0.2 * np.sin(u))], axis=-1
    ).astype(np.float32)
    path = tmp_path / f"_src_{asset_id}.hdr"
    assert cv2.imwrite(str(path), radiance)
    data = path.read_bytes()
    path.unlink()
    return data


class FakeResponse:
    def __init__(self, payload: Any = None, content: bytes = b"") -> None:
        self._payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload

    def iter_content(self, chunk_size: int = 1 << 16):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class FakeSession:
    """Serves the API fixtures and records every URL requested."""

    def __init__(self, blobs: dict[str, bytes], corrupt: set[str] | None = None) -> None:
        self.headers: dict[str, str] = {}
        self.blobs = blobs
        self.corrupt = corrupt or set()
        self.urls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.urls.append(url)
        if url.endswith("/assets?t=hdris"):
            return FakeResponse(payload=ASSETS_PAYLOAD)
        if "/files/" in url:
            return FakeResponse(payload=_files_payload(url.rsplit("/", 1)[-1]))
        if url in self.blobs:
            data = self.blobs[url]
            if url in self.corrupt:
                data = data + b"tampered"
            return FakeResponse(content=data)
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def blobs(tmp_path: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for asset_id in ASSETS_PAYLOAD:
        url = _files_payload(asset_id)["hdri"]["2k"]["hdr"]["url"]
        out[url] = _hdr_bytes(asset_id, tmp_path)
    return out


def make_client(session: FakeSession, **kwargs: Any) -> dl.PolyHavenClient:
    """A client wired to the fake session with a fake clock (no real sleeping)."""
    clock = {"t": 0.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    client = dl.PolyHavenClient(session=session, sleep=sleep, now=now, **kwargs)
    client._clock = clock  # type: ignore[attr-defined]
    return client


# --- pure helpers -----------------------------------------------------------


def test_pick_download_selects_the_requested_resolution() -> None:
    choice = dl.pick_download(_files_payload("x"), "2k")
    assert choice["url"].endswith("/hdr/2k/x_2k.hdr")
    assert choice["download_format"] == "hdr"
    assert choice["size"] == 6156727


@pytest.mark.parametrize("res", ["1k", "4k", "16k"])
def test_pick_download_honours_other_resolutions(res: str) -> None:
    assert dl.pick_download(_files_payload("x"), res)["url"].endswith(f"/{res}/x_{res}.hdr")


def test_pick_download_rejects_missing_resolution() -> None:
    with pytest.raises(LookupError, match="24k"):
        dl.pick_download(_files_payload("x"), "24k")


def test_pick_download_rejects_malformed_payload() -> None:
    with pytest.raises(LookupError, match="hdri"):
        dl.pick_download({"tonemapped": {"url": "u"}}, "2k")
    with pytest.raises(LookupError, match="format"):
        dl.pick_download({"hdri": {"2k": {"exr": {"url": "u"}}}}, "2k")


@pytest.mark.parametrize(
    "category,expected",
    [
        ("Desert & Arid/Karoo & Scrubland/Rocky Outcrops", "Desert & Arid"),
        ("Studio/Seamless/White", "Studio"),
        ("Industrial", "Industrial"),
        ("  Coast & Water/Beach  ", "Coast & Water"),
    ],
)
def test_primary_category_takes_the_top_segment(category: str, expected: str) -> None:
    assert dl.primary_category({"category": category}) == expected


@pytest.mark.parametrize("value", [None, "", "   ", 42, []])
def test_primary_category_falls_back_to_unknown(value: Any) -> None:
    assert dl.primary_category({"category": value}) == "unknown"


def _synthetic_hdr(height: int = 48, width: int = 96) -> np.ndarray:
    """A smooth, strictly positive BGR radiance field for tonemap tests."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u = xx / width * 2.0 * np.pi
    v = yy / height * np.pi
    base = 0.6 + 0.5 * np.sin(u) * np.cos(v) + 0.3 * np.cos(2.0 * v)
    base = np.clip(base, 0.05, None)
    return np.stack([base * 0.9, base, base * 1.1], axis=-1).astype(np.float32)


def test_luminance_uses_rec709_weights_in_bgr_order() -> None:
    # A pure-red pixel in BGR order is (0, 0, 1) and must weigh 0.2126.
    red = dl.luminance_bgr(np.array([[[0.0, 0.0, 1.0]]], np.float32))
    green = dl.luminance_bgr(np.array([[[0.0, 1.0, 0.0]]], np.float32))
    blue = dl.luminance_bgr(np.array([[[1.0, 0.0, 0.0]]], np.float32))
    assert red[0, 0] == pytest.approx(0.2126, abs=1e-6)
    assert green[0, 0] == pytest.approx(0.7152, abs=1e-6)
    assert blue[0, 0] == pytest.approx(0.0722, abs=1e-6)


def test_log_average_luminance_is_the_geometric_mean() -> None:
    lum = np.array([1.0, 4.0, 16.0], dtype=np.float32)
    assert dl.log_average_luminance(lum) == pytest.approx(4.0, rel=1e-5)


def _reference_log_average(lum: np.ndarray) -> float:
    """cos(latitude) weighted geometric mean, written out independently."""
    height = lum.shape[0]
    latitude = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    weights = np.broadcast_to(np.cos(latitude)[:, None], lum.shape)
    return float(np.exp((weights * np.log(lum + 1e-6)).sum() / weights.sum()))


def test_tonemap_reinhard_matches_the_formula() -> None:
    radiance = _synthetic_hdr()
    key = dl.DEFAULT_KEY

    lum = radiance.astype(np.float64) @ np.array([0.0722, 0.7152, 0.2126])
    l_avg = _reference_log_average(lum)
    scaled = key / l_avg * lum
    displayed = scaled / (1.0 + scaled)
    expected_linear = np.clip(radiance * (displayed / lum)[..., None], 0.0, 1.0)
    expected = np.clip(
        np.rint(np.power(expected_linear, 1 / 2.2) * 255.0), 0, 255
    ).astype(np.uint8)

    out = dl.tonemap_reinhard(radiance, key=key)
    assert out.dtype == np.uint8
    assert out.shape == radiance.shape
    assert np.abs(out.astype(int) - expected.astype(int)).max() <= 1


@pytest.mark.parametrize("factor", [0.01, 100.0])
def test_tonemap_reinhard_is_exposure_invariant(factor: float) -> None:
    """Scaling the whole scene must not change the tonemapped result."""
    radiance = _synthetic_hdr()
    reference = dl.tonemap_reinhard(radiance).astype(np.float64)
    scaled = dl.tonemap_reinhard(radiance * np.float32(factor)).astype(np.float64)
    mae = float(np.abs(scaled - reference).mean())
    assert mae < 1.0, f"x{factor} changed the output by {mae:.3f} levels"


def test_tonemap_reinhard_key_controls_brightness() -> None:
    radiance = _synthetic_hdr()
    dark = dl.tonemap_reinhard(radiance, key=0.06).astype(float).mean()
    mid = dl.tonemap_reinhard(radiance, key=0.18).astype(float).mean()
    bright = dl.tonemap_reinhard(radiance, key=0.50).astype(float).mean()
    assert dark < mid < bright


def test_tonemap_reinhard_is_monotonic_and_clamps_bad_values() -> None:
    radiance = np.linspace(0.0, 50.0, 256, dtype=np.float32).reshape(-1, 1, 1)
    out = dl.tonemap_reinhard(np.repeat(radiance, 3, axis=2))
    assert np.all(np.diff(out[:, 0, 0].astype(int)) >= 0)

    nasty = np.array([[[np.nan, -5.0, np.inf]]], dtype=np.float32)
    clamped = dl.tonemap_reinhard(nasty)
    assert np.isfinite(clamped).all()
    assert (clamped == 0).all()


def test_tonemap_reinhard_rejects_bad_arguments() -> None:
    radiance = _synthetic_hdr(4, 4)
    with pytest.raises(ValueError):
        dl.tonemap_reinhard(None)
    with pytest.raises(ValueError):
        dl.tonemap_reinhard(radiance, key=0.0)
    with pytest.raises(ValueError):
        dl.tonemap_reinhard(radiance, gamma=0.0)
    with pytest.raises(ValueError, match="BGR"):
        dl.tonemap_reinhard(np.zeros((4, 4), dtype=np.float32))


def test_md5_helpers_agree(tmp_path: Path) -> None:
    data = b"poly haven" * 1000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert dl.md5_of_file(path) == dl.md5_of_bytes(data) == hashlib.md5(data).hexdigest()


def test_select_asset_ids_is_alphabetical_and_limited() -> None:
    ids = dl.select_asset_ids(ASSETS_PAYLOAD, None)
    assert ids == ["alpha_asset", "beta_asset", "gamma_asset"]
    assert dl.select_asset_ids(ASSETS_PAYLOAD, 2) == ["alpha_asset", "beta_asset"]
    assert dl.select_asset_ids(ASSETS_PAYLOAD, 0) == []


def test_default_manifest_path_follows_spec_layout() -> None:
    assert dl._default_manifest_path(
        Path("data_root/raw/polyhaven")
    ) == Path("data_root/manifest.json")


# --- client behaviour -------------------------------------------------------


def test_client_sets_identifiable_user_agent(blobs: dict[str, bytes]) -> None:
    session = FakeSession(blobs)
    client = make_client(session)
    assert session.headers["User-Agent"] == "fisheye-calib-research/0.1"
    assert client.user_agent == session.headers["User-Agent"]
    assert not hasattr(dl, "REPO_URL")


def test_client_throttles_between_requests(blobs: dict[str, bytes]) -> None:
    """SPEC.md 3.7: at least 0.2 s between requests."""
    session = FakeSession(blobs)
    client = make_client(session)
    stamps: list[float] = []
    for _ in range(4):
        client.list_hdris()
        stamps.append(client._clock["t"])  # type: ignore[attr-defined]
    gaps = np.diff(stamps)
    assert gaps.min() >= dl.MIN_REQUEST_INTERVAL - 1e-12, gaps
    assert client.request_count == 4


def test_client_rejects_too_short_an_interval(blobs: dict[str, bytes]) -> None:
    with pytest.raises(ValueError, match="min_interval"):
        make_client(FakeSession(blobs), min_interval=0.05)


def test_download_writes_atomically_and_returns_md5(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    session = FakeSession(blobs)
    client = make_client(session)
    url, data = next(iter(blobs.items()))
    dest = tmp_path / "nested" / "file.hdr"

    digest = client.download(url, dest)
    assert dest.read_bytes() == data
    assert digest == dl.md5_of_bytes(data)
    assert not list(tmp_path.rglob("*.part")), "temporary file left behind"


# --- end-to-end run ---------------------------------------------------------


def run_download(
    tmp_path: Path,
    blobs: dict[str, bytes],
    *,
    limit: int | None = None,
    corrupt: set[str] | None = None,
    session: FakeSession | None = None,
    key: float = dl.DEFAULT_KEY,
    retonemap: bool = False,
) -> tuple[dict[str, Any], FakeSession, Path, Path]:
    out_dir = tmp_path / "data_root" / "raw" / "polyhaven"
    manifest_path = tmp_path / "data_root" / "manifest.json"
    session = session if session is not None else FakeSession(blobs, corrupt=corrupt)
    manifest = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        limit=limit,
        key=key,
        retonemap=retonemap,
        client=make_client(session),
        log=lambda _msg: None,
    )
    return manifest, session, out_dir, manifest_path


def use_published_md5(blobs: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
    """Make pick_download report the real blob MD5, as the live API does."""
    real_pick = dl.pick_download

    def patched(files: dict[str, Any], res: str) -> dict[str, Any]:
        choice = real_pick(files, res)
        choice["md5"] = dl.md5_of_bytes(blobs[choice["url"]])
        return choice

    monkeypatch.setattr(dl, "pick_download", patched)


def test_run_downloads_converts_and_writes_manifest(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    manifest, session, out_dir, manifest_path = run_download(tmp_path, blobs)

    assert manifest["count"] == 3
    assert manifest["res"] == "2k"
    assert manifest["download_format"] == "hdr"
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["count"] == 3

    # Rows are in alphabetical asset order and each JPG exists.
    ids = [row["asset_id"] for row in manifest["assets"]]
    assert ids == ["alpha_asset", "beta_asset", "gamma_asset"]
    for row in manifest["assets"]:
        jpg = out_dir / row["file"]
        assert jpg.is_file()
        assert row["file"] == f"{row['asset_id']}.jpg"
        assert dl.md5_of_file(jpg) == row["jpg_md5"]

    # The source .hdr is kept alongside each JPG (needed by --retonemap).
    assert sorted(p.suffix for p in out_dir.iterdir()) == [".hdr"] * 3 + [".jpg"] * 3
    for row in manifest["assets"]:
        source = out_dir / row["source_file"]
        assert source.is_file()
        assert dl.md5_of_file(source) == row["md5"]


def test_manifest_rows_carry_every_spec_field(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    manifest, _, _, _ = run_download(tmp_path, blobs)
    row = next(r for r in manifest["assets"] if r["asset_id"] == "alpha_asset")

    for key in ("asset_id", "categories", "width", "height", "md5",
                "source_url", "download_format"):
        assert key in row, key

    assert row["categories"] == ASSETS_PAYLOAD["alpha_asset"]["categories"]
    assert row["primary_category"] == "Studio"
    assert row["category"] == "Studio/Seamless/White"
    assert row["download_format"] == "hdr"
    assert row["source_url"].endswith("/hdr/2k/alpha_asset_2k.hdr")
    assert (row["width"], row["height"]) == (HDR_W, HDR_H)
    assert row["md5"] == dl.md5_of_bytes(blobs[row["source_url"]])


def test_stored_jpg_matches_the_tonemapped_source(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    manifest, _, out_dir, _ = run_download(tmp_path, blobs)
    row = manifest["assets"][0]

    source = blobs[row["source_url"]]
    src_path = tmp_path / "ref.hdr"
    src_path.write_bytes(source)
    radiance = cv2.imread(str(src_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    expected = dl.tonemap_reinhard(radiance)

    stored = cv2.imread(str(out_dir / row["file"]))
    assert stored.shape == expected.shape
    # JPEG is lossy at quality 95, so compare statistically rather than exactly.
    assert np.abs(stored.astype(int) - expected.astype(int)).mean() < 6.0


def test_run_respects_limit(tmp_path: Path, blobs: dict[str, bytes]) -> None:
    manifest, session, out_dir, _ = run_download(tmp_path, blobs, limit=2)
    assert [row["asset_id"] for row in manifest["assets"]] == [
        "alpha_asset",
        "beta_asset",
    ]
    assert len(list(out_dir.glob("*.jpg"))) == 2
    assert not any("gamma_asset" in url for url in session.urls)


def test_resume_skips_cached_assets(tmp_path: Path, blobs: dict[str, bytes]) -> None:
    first, _, out_dir, manifest_path = run_download(tmp_path, blobs)
    jpg_md5s = {row["asset_id"]: row["jpg_md5"] for row in first["assets"]}

    second_session = FakeSession(blobs)
    second = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        client=make_client(second_session),
        log=lambda _msg: None,
    )

    # Only the asset listing is fetched; no /files/ and no blob downloads.
    assert len(second_session.urls) == 1
    assert second_session.urls[0].endswith("/assets?t=hdris")
    assert second["count"] == 3
    assert {row["asset_id"]: row["jpg_md5"] for row in second["assets"]} == jpg_md5s


def test_resume_redownloads_when_the_jpg_is_altered(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    first, _, out_dir, manifest_path = run_download(tmp_path, blobs)
    victim = out_dir / first["assets"][0]["file"]
    victim.write_bytes(victim.read_bytes() + b"\x00corrupted")

    session = FakeSession(blobs)
    dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        client=make_client(session),
        log=lambda _msg: None,
    )
    assert any("alpha_asset" in url and url.endswith(".hdr") for url in session.urls)
    assert not any("beta_asset" in url and url.endswith(".hdr") for url in session.urls)


def test_resume_redownloads_when_the_jpg_is_missing(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    first, _, out_dir, manifest_path = run_download(tmp_path, blobs)
    (out_dir / first["assets"][2]["file"]).unlink()

    session = FakeSession(blobs)
    manifest = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        client=make_client(session),
        log=lambda _msg: None,
    )
    assert any("gamma_asset" in url and url.endswith(".hdr") for url in session.urls)
    assert manifest["count"] == 3


# --- MD5 verification -------------------------------------------------------


def test_md5_mismatch_rejects_the_asset(
    tmp_path: Path, blobs: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tampered download must not produce a JPG or a manifest row."""
    use_published_md5(blobs, monkeypatch)

    bad_url = _files_payload("beta_asset")["hdri"]["2k"]["hdr"]["url"]
    manifest, _, out_dir, _ = run_download(tmp_path, blobs, corrupt={bad_url})

    ids = [row["asset_id"] for row in manifest["assets"]]
    assert ids == ["alpha_asset", "gamma_asset"]
    assert not (out_dir / "beta_asset.jpg").exists()
    assert not list(out_dir.glob("beta_asset*.hdr"))
    # The assets that verified fine keep their sources.
    assert sorted(p.name for p in out_dir.glob("*.hdr")) == [
        "alpha_asset_2k.hdr",
        "gamma_asset_2k.hdr",
    ]


def test_md5_match_accepts_the_asset(
    tmp_path: Path, blobs: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    use_published_md5(blobs, monkeypatch)
    manifest, _, _, _ = run_download(tmp_path, blobs)
    assert manifest["count"] == 3


def test_failed_asset_does_not_abort_the_run(
    tmp_path: Path, blobs: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_pick = dl.pick_download

    def pick_failing_beta(files: dict[str, Any], res: str) -> dict[str, Any]:
        choice = real_pick(files, res)
        if "beta_asset" in choice["url"]:
            raise LookupError("simulated missing resolution")
        return choice

    monkeypatch.setattr(dl, "pick_download", pick_failing_beta)
    manifest, _, out_dir, _ = run_download(tmp_path, blobs)
    assert [row["asset_id"] for row in manifest["assets"]] == [
        "alpha_asset",
        "gamma_asset",
    ]
    assert not (out_dir / "beta_asset.jpg").exists()


# --- manifest round-trip ----------------------------------------------------


def test_load_manifest_tolerates_missing_and_broken_files(tmp_path: Path) -> None:
    assert dl.load_manifest(tmp_path / "nope.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert dl.load_manifest(broken) == {}
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"assets": "nope"}', encoding="utf-8")
    assert dl.load_manifest(wrong) == {}


def test_load_manifest_reads_what_run_wrote(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    manifest, _, _, manifest_path = run_download(tmp_path, blobs)
    loaded = dl.load_manifest(manifest_path)
    assert set(loaded) == {row["asset_id"] for row in manifest["assets"]}
    assert loaded["alpha_asset"]["primary_category"] == "Studio"


# --- CLI --------------------------------------------------------------------


def test_cli_parses_the_spec_invocation() -> None:
    args = dl.build_parser().parse_args(
        ["--out", "data_root/raw/polyhaven", "--res", "2k", "--limit", "5"]
    )
    assert args.out == "data_root/raw/polyhaven"
    assert args.res == "2k"
    assert args.limit == 5
    assert args.manifest is None


def test_main_runs_end_to_end(tmp_path: Path, blobs: dict[str, bytes]) -> None:
    out_dir = tmp_path / "data_root" / "raw" / "polyhaven"
    session = FakeSession(blobs)
    code = dl.main(
        ["--out", str(out_dir), "--res", "2k", "--limit", "2"],
        client=make_client(session),
    )
    assert code == 0
    assert (tmp_path / "data_root" / "manifest.json").is_file()
    assert sorted(p.name for p in out_dir.glob("*.jpg")) == [
        "alpha_asset.jpg",
        "beta_asset.jpg",
    ]


# --- keeping the source, and --retonemap ------------------------------------


def test_existing_source_is_reused_instead_of_downloaded(
    tmp_path: Path, blobs: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified .hdr already on disk must not be fetched again."""
    use_published_md5(blobs, monkeypatch)
    _, _, out_dir, manifest_path = run_download(tmp_path, blobs)

    # Drop the JPGs so the cache check cannot short-circuit the run.
    for jpg in out_dir.glob("*.jpg"):
        jpg.unlink()

    session = FakeSession(blobs)
    manifest = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        client=make_client(session),
        log=lambda _msg: None,
    )
    assert manifest["count"] == 3
    assert len(list(out_dir.glob("*.jpg"))) == 3
    # /files/ is still consulted to learn the MD5, but no blob is transferred.
    assert not any(url.endswith(".hdr") for url in session.urls)


def test_retonemap_rebuilds_jpgs_without_any_download(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    first, _, out_dir, manifest_path = run_download(tmp_path, blobs)
    before = {row["asset_id"]: row["jpg_md5"] for row in first["assets"]}

    session = FakeSession(blobs)
    second = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        key=0.45,
        retonemap=True,
        client=make_client(session),
        log=lambda _msg: None,
    )

    # Only the asset listing is fetched: no /files/ and no blobs.
    assert len(session.urls) == 1
    assert session.urls[0].endswith("/assets?t=hdris")

    after = {row["asset_id"]: row["jpg_md5"] for row in second["assets"]}
    assert set(after) == set(before)
    assert after != before, "a different key must change the stored JPGs"
    assert second["tonemap"]["key"] == pytest.approx(0.45)

    brightest = cv2.imread(str(out_dir / second["assets"][0]["file"]))
    assert brightest is not None


def test_retonemap_falls_back_to_download_when_source_is_gone(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    first, _, out_dir, manifest_path = run_download(tmp_path, blobs)
    (out_dir / first["assets"][1]["source_file"]).unlink()

    session = FakeSession(blobs)
    manifest = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        retonemap=True,
        client=make_client(session),
        log=lambda _msg: None,
    )
    assert manifest["count"] == 3
    downloaded = [url for url in session.urls if url.endswith(".hdr")]
    assert len(downloaded) == 1 and "beta_asset" in downloaded[0]


def test_retonemap_falls_back_when_the_source_md5_no_longer_matches(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    first, _, out_dir, manifest_path = run_download(tmp_path, blobs)
    victim = out_dir / first["assets"][0]["source_file"]
    victim.write_bytes(victim.read_bytes() + b"\x00tampered")

    session = FakeSession(blobs)
    dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        retonemap=True,
        client=make_client(session),
        log=lambda _msg: None,
    )
    downloaded = [url for url in session.urls if url.endswith(".hdr")]
    assert len(downloaded) == 1 and "alpha_asset" in downloaded[0]


# --- incremental, merging manifest ------------------------------------------


def test_manifest_survives_a_keyboard_interrupt(
    tmp_path: Path, blobs: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) Work finished before an interrupt stays recorded and is then skipped."""
    out_dir = tmp_path / "data_root" / "raw" / "polyhaven"
    manifest_path = tmp_path / "data_root" / "manifest.json"
    real_download = dl.download_asset

    def interrupt_on_gamma(client, asset_id, *args, **kwargs):
        if asset_id == "gamma_asset":
            raise KeyboardInterrupt
        return real_download(client, asset_id, *args, **kwargs)

    monkeypatch.setattr(dl, "download_asset", interrupt_on_gamma)
    with pytest.raises(KeyboardInterrupt):
        dl.run(
            out_dir=out_dir,
            manifest_path=manifest_path,
            res="2k",
            client=make_client(FakeSession(blobs)),
            log=lambda _msg: None,
        )

    saved = dl.load_manifest(manifest_path)
    assert set(saved) == {"alpha_asset", "beta_asset"}
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["count"] == 2

    monkeypatch.setattr(dl, "download_asset", real_download)
    session = FakeSession(blobs)
    manifest = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        client=make_client(session),
        log=lambda _msg: None,
    )
    assert manifest["count"] == 3
    # The two completed assets are not fetched again; only gamma is.
    downloaded = [url for url in session.urls if url.endswith(".hdr")]
    assert len(downloaded) == 1 and "gamma_asset" in downloaded[0]


def test_limited_run_does_not_drop_existing_records(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    """(b) A later --limit 2 run must leave the manifest at three assets."""
    full, _, out_dir, manifest_path = run_download(tmp_path, blobs)
    assert full["count"] == 3

    session = FakeSession(blobs)
    limited = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        limit=2,
        client=make_client(session),
        log=lambda _msg: None,
    )
    assert limited["count"] == 3
    assert [row["asset_id"] for row in limited["assets"]] == [
        "alpha_asset",
        "beta_asset",
        "gamma_asset",
    ]
    assert dl.load_manifest(manifest_path).keys() == {
        "alpha_asset",
        "beta_asset",
        "gamma_asset",
    }


def test_manifest_is_written_after_every_asset(
    tmp_path: Path, blobs: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "data_root" / "raw" / "polyhaven"
    manifest_path = tmp_path / "data_root" / "manifest.json"
    counts: list[int] = []
    real_download = dl.download_asset

    def record(client, asset_id, *args, **kwargs):
        counts.append(len(dl.load_manifest(manifest_path)))
        return real_download(client, asset_id, *args, **kwargs)

    monkeypatch.setattr(dl, "download_asset", record)
    dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        client=make_client(FakeSession(blobs)),
        log=lambda _msg: None,
    )
    # Before each asset the manifest already holds everything done so far.
    assert counts == [0, 1, 2]


def test_write_manifest_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    dl.write_manifest(path, dl.build_manifest({}, "2k"))
    assert json.loads(path.read_text(encoding="utf-8"))["count"] == 0
    assert not list(tmp_path.glob("*.tmp")), "temporary file left behind"


def test_build_manifest_orders_rows_by_asset_id() -> None:
    rows = {
        "zulu": {"asset_id": "zulu"},
        "alpha": {"asset_id": "alpha"},
        "mike": {"asset_id": "mike"},
    }
    manifest = dl.build_manifest(rows, "2k")
    assert [row["asset_id"] for row in manifest["assets"]] == ["alpha", "mike", "zulu"]
    assert manifest["tonemap"]["operator"] == "reinhard2002"


def test_cli_exposes_key_and_retonemap() -> None:
    args = dl.build_parser().parse_args(
        ["--out", "o", "--key", "0.3", "--retonemap"]
    )
    assert args.key == pytest.approx(0.3)
    assert args.retonemap is True

    default = dl.build_parser().parse_args(["--out", "o"])
    assert default.key == pytest.approx(dl.DEFAULT_KEY)
    assert default.retonemap is False


# --- cos(latitude) weighting ------------------------------------------------


def test_latitude_weights_follow_the_panorama_convention() -> None:
    """Row i sits at latitude (0.5 - (i + 0.5) / H) * pi (SPEC.md 3.1)."""
    height = 8
    weights = dl.latitude_weights(height)
    expected = np.cos((0.5 - (np.arange(height) + 0.5) / height) * np.pi)
    assert np.allclose(weights, expected, atol=1e-12)

    # Symmetric about the equator, smallest at the poles, largest in the middle.
    assert np.allclose(weights, weights[::-1], atol=1e-12)
    assert weights[0] == weights.min()
    assert weights[height // 2] == weights.max()
    assert (weights > 0).all()


def test_log_average_luminance_discounts_a_bright_sky() -> None:
    """A bright band across the top 10% of rows must count for less than its
    pixel share, because those rows cover very little solid angle."""
    height, width = 100, 20
    lum = np.full((height, width), 0.1, dtype=np.float64)
    lum[: height // 10] = 50.0

    weighted = dl.log_average_luminance(lum)
    unweighted = float(np.exp(np.mean(np.log(lum + dl.LOG_EPS))))

    assert weighted < unweighted
    # The bright band holds 10% of the pixels but only ~2.4% of the solid
    # angle, so discounting it moves the average substantially, not marginally.
    assert weighted < 0.7 * unweighted


def test_log_average_luminance_is_unchanged_for_a_flat_panorama() -> None:
    """Weighting must not bias a uniform scene."""
    lum = np.full((64, 128), 2.5, dtype=np.float64)
    assert dl.log_average_luminance(lum) == pytest.approx(2.5, rel=1e-5)


def test_log_average_luminance_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        dl.log_average_luminance(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        dl.latitude_weights(0)


def test_weighted_average_changes_the_tonemap_of_a_sky_heavy_scene() -> None:
    """The whole point: a bright sky no longer drags the exposure down."""
    height, width = 100, 200
    base = np.full((height, width), 0.1, dtype=np.float32)
    base[: height // 10] = 50.0
    radiance = np.repeat(base[:, :, None], 3, axis=2)

    out = dl.tonemap_reinhard(radiance)
    ground = out[height // 2 :, :, 0].astype(float).mean()

    # With an unweighted average the sky would pull L_avg up and crush the
    # ground to near-black; weighted, the ground stays readable.
    assert ground > 40.0


# --- tonemap_key on each row ------------------------------------------------


def test_manifest_rows_record_the_tonemap_key(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    manifest, _, _, _ = run_download(tmp_path, blobs, key=0.18)
    assert manifest["assets"]
    for row in manifest["assets"]:
        assert row["tonemap_key"] == pytest.approx(0.18)


def test_changing_the_key_rebuilds_jpgs_without_retonemap(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    """A different --key invalidates the cache even though the MD5 matches."""
    first, _, out_dir, manifest_path = run_download(tmp_path, blobs, key=0.18)
    before = {row["asset_id"]: row["jpg_md5"] for row in first["assets"]}
    assert all(row["tonemap_key"] == pytest.approx(0.18) for row in first["assets"])

    session = FakeSession(blobs)
    second = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        key=0.25,
        client=make_client(session),
        log=lambda _msg: None,
    )

    after = {row["asset_id"]: row["jpg_md5"] for row in second["assets"]}
    assert set(after) == set(before)
    assert all(row["tonemap_key"] == pytest.approx(0.25) for row in second["assets"])
    assert all(after[k] != before[k] for k in before), "JPGs were not regenerated"

    # Rebuilt from the local .hdr: nothing was downloaded.
    assert not any(url.endswith(".hdr") for url in session.urls)


def test_same_key_still_skips_cached_assets(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    first, _, out_dir, manifest_path = run_download(tmp_path, blobs, key=0.22)
    before = {row["asset_id"]: row["jpg_md5"] for row in first["assets"]}

    session = FakeSession(blobs)
    second = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        key=0.22,
        client=make_client(session),
        log=lambda _msg: None,
    )
    assert len(session.urls) == 1, "an unchanged key must not reprocess anything"
    assert {row["asset_id"]: row["jpg_md5"] for row in second["assets"]} == before


def test_key_change_falls_back_to_download_without_a_local_source(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    first, _, out_dir, manifest_path = run_download(tmp_path, blobs, key=0.18)
    (out_dir / first["assets"][1]["source_file"]).unlink()

    session = FakeSession(blobs)
    second = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        key=0.25,
        client=make_client(session),
        log=lambda _msg: None,
    )
    assert all(row["tonemap_key"] == pytest.approx(0.25) for row in second["assets"])
    downloaded = [url for url in session.urls if url.endswith(".hdr")]
    assert len(downloaded) == 1 and "beta_asset" in downloaded[0]


def test_rows_without_a_tonemap_key_are_rebuilt(
    tmp_path: Path, blobs: dict[str, bytes]
) -> None:
    """A manifest from before the field existed has unknown provenance."""
    first, _, out_dir, manifest_path = run_download(tmp_path, blobs)

    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in legacy["assets"]:
        row.pop("tonemap_key")
    manifest_path.write_text(json.dumps(legacy), encoding="utf-8")

    session = FakeSession(blobs)
    second = dl.run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res="2k",
        client=make_client(session),
        log=lambda _msg: None,
    )
    assert all("tonemap_key" in row for row in second["assets"])
    assert not any(url.endswith(".hdr") for url in session.urls)


def test_tonemap_key_changed_helper() -> None:
    assert dl._tonemap_key_changed(None, 0.18) is False
    assert dl._tonemap_key_changed({"tonemap_key": 0.18}, 0.18) is False
    assert dl._tonemap_key_changed({"tonemap_key": 0.25}, 0.18) is True
    assert dl._tonemap_key_changed({}, 0.18) is True
    assert dl._tonemap_key_changed({"tonemap_key": "nonsense"}, 0.18) is True
