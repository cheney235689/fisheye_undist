"""Download Poly Haven HDRI panoramas and write ``manifest.json``.

SPEC.md 3.7.  Poly Haven's ``/files/{id}`` endpoint was queried before writing
this module; what it actually returns is::

    {"hdri": {"1k": {"exr": {url, md5, size}, "hdr": {...}}, "2k": {...}, ...},
     "tonemapped": {url, md5, size},
     "colorchart": {url, md5, size}}

The important finding is that ``tonemapped`` has **no per-resolution
variants** -- it is a single full-resolution JPG (5-62 MB, 16k-29k wide).  So
SPEC.md 3.7's preferred "tonemapped JPG at that resolution" does not exist and
every asset takes the documented fallback: fetch ``hdri/{res}/hdr`` (6.2 MB at
2k, MD5 published by the API), tonemap locally and store an 8-bit JPG.

The source ``.hdr`` is kept next to the JPG, which makes two things cheap:
re-running after an interruption skips any file already on disk with the right
MD5, and ``--retonemap`` re-derives the JPGs from the local ``.hdr`` files
without downloading anything when the tonemapping parameters change.

The manifest is rewritten atomically after every asset and merged with
whatever it already held, so an interrupted run keeps its completed work and a
``--limit`` run never drops records it did not touch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np
import requests

API_BASE = "https://api.polyhaven.com"
USER_AGENT = "fisheye-calib-research/0.1"

MIN_REQUEST_INTERVAL = 0.2  # seconds, SPEC.md 3.7
DEFAULT_RES = "2k"
DEFAULT_JPEG_QUALITY = 95
DEFAULT_KEY = 0.18
DEFAULT_GAMMA = 2.2
DOWNLOAD_FORMAT = "hdr"
CHUNK_SIZE = 1 << 16

# Rec.709 luminance weights, in OpenCV's BGR channel order.
LUMA_BGR = (0.0722, 0.7152, 0.2126)
LOG_EPS = 1e-6

__all__ = [
    "PolyHavenClient",
    "ManifestEntry",
    "pick_download",
    "primary_category",
    "luminance_bgr",
    "latitude_weights",
    "log_average_luminance",
    "tonemap_reinhard",
    "md5_of_file",
    "md5_of_bytes",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "select_asset_ids",
    "download_asset",
    "retonemap_asset",
    "run",
    "main",
]


# --- HTTP ------------------------------------------------------------------


class PolyHavenClient:
    """Thin, rate-limited wrapper over the Poly Haven API.

    ``sleep`` and ``now`` are injectable so tests can drive the throttle with a
    fake clock instead of really waiting.
    """

    def __init__(
        self,
        session: Any | None = None,
        *,
        user_agent: str = USER_AGENT,
        min_interval: float = MIN_REQUEST_INTERVAL,
        api_base: str = API_BASE,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_interval < MIN_REQUEST_INTERVAL:
            raise ValueError(
                f"min_interval must be >= {MIN_REQUEST_INTERVAL} s (SPEC.md 3.7), "
                f"got {min_interval}"
            )
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.user_agent = user_agent
        self.min_interval = float(min_interval)
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self._sleep = sleep
        self._now = now
        self._last_request: float | None = None
        self.request_count = 0

    def _throttle(self) -> None:
        if self._last_request is not None:
            elapsed = self._now() - self._last_request
            if elapsed < self.min_interval:
                self._sleep(self.min_interval - elapsed)
        self._last_request = self._now()
        self.request_count += 1

    def get_json(self, path: str) -> Any:
        """GET an API path (e.g. ``/assets?t=hdris``) and parse the JSON body."""
        self._throttle()
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def download(self, url: str, dest: Path) -> str:
        """Stream ``url`` into ``dest``, returning the MD5 of what was written.

        The bytes land in a ``.part`` sibling first so an interrupted run never
        leaves a short file that looks complete.
        """
        self._throttle()
        digest = hashlib.md5()
        dest = Path(dest)
        tmp = dest.with_name(dest.name + ".part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        response = self.session.get(url, timeout=self.timeout, stream=True)
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
        os.replace(tmp, dest)
        return digest.hexdigest()

    def list_hdris(self) -> dict[str, Any]:
        """All HDRI assets, keyed by ``asset_id``."""
        return self.get_json("/assets?t=hdris")

    def asset_files(self, asset_id: str) -> dict[str, Any]:
        return self.get_json(f"/files/{asset_id}")


# --- pure helpers ----------------------------------------------------------


def md5_of_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def md5_of_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_download(files: dict[str, Any], res: str) -> dict[str, Any]:
    """Choose the file to fetch for ``res`` from a ``/files/{id}`` payload.

    Returns a dict with ``url``, ``md5``, ``size`` and ``download_format``.
    Raises :class:`LookupError` if the resolution or format is absent, so a
    malformed or unexpected payload surfaces instead of being guessed at.
    """
    hdri = files.get("hdri")
    if not isinstance(hdri, dict):
        raise LookupError("payload has no 'hdri' section")
    by_format = hdri.get(res)
    if not isinstance(by_format, dict):
        raise LookupError(f"resolution {res!r} not available; have {sorted(hdri)}")
    entry = by_format.get(DOWNLOAD_FORMAT)
    if not isinstance(entry, dict) or "url" not in entry:
        raise LookupError(
            f"format {DOWNLOAD_FORMAT!r} not available at {res!r}; "
            f"have {sorted(by_format)}"
        )
    return {
        "url": entry["url"],
        "md5": entry.get("md5"),
        "size": entry.get("size"),
        "download_format": DOWNLOAD_FORMAT,
    }


def primary_category(asset: dict[str, Any]) -> str:
    """Top segment of the hierarchical ``category`` string.

    Poly Haven gives every asset exactly one ``category`` path such as
    ``"Desert & Arid/Karoo & Scrubland/Rocky Outcrops"``; its first segment is
    the 12-way scene class used for stratified splitting (SPEC.md 4.10).
    """
    raw = asset.get("category")
    if not isinstance(raw, str) or not raw.strip():
        return "unknown"
    return raw.split("/")[0].strip() or "unknown"


# --- tonemapping -----------------------------------------------------------


def luminance_bgr(image: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of a BGR radiance image.

    ``L = 0.2126 R + 0.7152 G + 0.0722 B``; OpenCV hands us BGR, so the weights
    are applied in that order.
    """
    radiance = np.asarray(image, dtype=np.float32)
    if radiance.ndim != 3 or radiance.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) BGR image, got {radiance.shape}")
    weights = np.asarray(LUMA_BGR, dtype=np.float32)
    return radiance @ weights


def latitude_weights(height: int) -> np.ndarray:
    """``cos(lat)`` per panorama row, the solid angle each row subtends.

    Row ``i`` of an ``H``-row equirectangular image sits at latitude
    ``(0.5 - (i + 0.5) / H) * pi`` (SPEC.md 3.1), so rows near the poles cover
    far less of the sphere than rows near the horizon and must count for less.
    """
    height = int(height)
    if height < 1:
        raise ValueError(f"height must be >= 1, got {height}")
    rows = np.arange(height, dtype=np.float64)
    latitude = (0.5 - (rows + 0.5) / height) * np.pi
    # Latitude stays strictly inside (-pi/2, pi/2), so cos is positive; the
    # clip only guards against round-off at the very first and last row.
    return np.clip(np.cos(latitude), 0.0, None)


def log_average_luminance(luminance: np.ndarray, eps: float = LOG_EPS) -> float:
    """Geometric mean brightness, weighted by ``cos(latitude)``.

    An equirectangular panorama massively over-samples the poles: the top and
    bottom rows are the same handful of directions smeared across the full
    width.  An unweighted average therefore lets a bright sky (or a bright
    ceiling) dominate the exposure far beyond the solid angle it really
    occupies.  Weighting each row by ``cos(latitude)`` makes the average an
    honest mean over the sphere.

    A 1-D input is treated as a single column of rows.
    """
    lum = np.asarray(luminance, dtype=np.float64)
    if lum.ndim == 1:
        lum = lum[:, None]
    if lum.ndim != 2:
        raise ValueError(f"expected a 2-D luminance map, got shape {lum.shape}")

    weights = np.broadcast_to(latitude_weights(lum.shape[0])[:, None], lum.shape)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("latitude weights sum to zero")
    return float(np.exp(float((weights * np.log(lum + eps)).sum()) / total))


def tonemap_reinhard(
    hdr: np.ndarray, key: float = DEFAULT_KEY, gamma: float = DEFAULT_GAMMA
) -> np.ndarray:
    """Reinhard 2002 global tonemap, then gamma encoding, to 8-bit BGR.

    Following "Photographic Tone Reproduction for Digital Images" (Reinhard et
    al. 2002), section 3:

        L     = 0.2126 R + 0.7152 G + 0.0722 B
        L_avg = exp(weighted_mean(log(L + 1e-6)))      # cos(latitude) weighted
        L_s   = (key / L_avg) * L
        L_d   = L_s / (1 + L_s)
        C_out = C * (L_d / L)

    The operator is **exposure invariant**: scaling the input by any positive
    factor scales ``L_avg`` by the same factor, so ``L_s`` -- and therefore the
    output -- is unchanged.  That is the point of the log-average step, and it
    is what the old naive ``C/(1+C)`` lacked.

    ``L_d / L`` is never evaluated as a quotient.  Substituting ``L_s`` gives

        L_d / L = (key / L_avg) / (1 + L_s)

    which is finite and correct as ``L -> 0`` (it tends to ``key / L_avg``), so
    black pixels need no special case and no epsilon creeps into the colour.
    """
    if hdr is None:
        raise ValueError("hdr image is None")
    if key <= 0.0:
        raise ValueError(f"key must be positive, got {key}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    radiance = np.asarray(hdr, dtype=np.float32)
    if radiance.ndim != 3 or radiance.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) BGR image, got {radiance.shape}")
    radiance = np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(radiance, 0.0, None, out=radiance)

    luminance = luminance_bgr(radiance)
    l_avg = log_average_luminance(luminance)

    exposure = float(key) / l_avg
    scaled = exposure * luminance.astype(np.float64)
    # C_out = C * L_d / L, with the quotient resolved analytically.
    ratio = (exposure / (1.0 + scaled)).astype(np.float32)

    mapped = np.clip(radiance * ratio[..., None], 0.0, 1.0)
    encoded = np.power(mapped, 1.0 / float(gamma))
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


# --- manifest --------------------------------------------------------------


@dataclass
class ManifestEntry:
    """One row of ``manifest.json`` (SPEC.md 3.7 plus the fields we resolved)."""

    asset_id: str
    categories: list[str]
    width: int
    height: int
    md5: str
    source_url: str
    download_format: str
    jpg_md5: str
    file: str
    source_file: str
    tonemap_key: float
    category: str
    primary_category: str
    name: str = ""
    authors: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "categories": list(self.categories),
            "category": self.category,
            "primary_category": self.primary_category,
            "attributes": dict(self.attributes),
            "authors": dict(self.authors),
            "width": int(self.width),
            "height": int(self.height),
            "md5": self.md5,
            "jpg_md5": self.jpg_md5,
            "source_url": self.source_url,
            "download_format": self.download_format,
            "file": self.file,
            "source_file": self.source_file,
            "tonemap_key": float(self.tonemap_key),
        }


def build_manifest(
    rows: Mapping[str, dict[str, Any]] | Iterable[dict[str, Any]],
    res: str,
    user_agent: str = USER_AGENT,
    key: float = DEFAULT_KEY,
) -> dict[str, Any]:
    """Assemble the manifest document, with rows in stable ``asset_id`` order."""
    if isinstance(rows, Mapping):
        entries = [rows[k] for k in sorted(rows)]
    else:
        entries = sorted(rows, key=lambda row: row.get("asset_id", ""))
    return {
        "api_base": API_BASE,
        "res": res,
        "download_format": DOWNLOAD_FORMAT,
        "user_agent": user_agent,
        "tonemap": {"operator": "reinhard2002", "key": float(key), "gamma": DEFAULT_GAMMA},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "count": len(entries),
        "assets": entries,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest atomically, so a crash cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Existing manifest rows keyed by ``asset_id``, or empty if absent/broken."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    assets = payload.get("assets") if isinstance(payload, dict) else payload
    if not isinstance(assets, list):
        return {}
    return {
        row["asset_id"]: row
        for row in assets
        if isinstance(row, dict) and "asset_id" in row
    }


def select_asset_ids(assets: dict[str, Any], limit: int | None) -> list[str]:
    """Alphabetical order, so ``--limit N`` is deterministic across runs."""
    ids = sorted(assets)
    return ids if limit is None else ids[: max(0, int(limit))]


# --- per-asset work --------------------------------------------------------


def source_path(out_dir: Path, asset_id: str, res: str, fmt: str = DOWNLOAD_FORMAT) -> Path:
    return Path(out_dir) / f"{asset_id}_{res}.{fmt}"


def _tonemap_key_changed(row: dict[str, Any] | None, key: float) -> bool:
    """True when the stored JPG was not produced with ``key``.

    A row written before ``tonemap_key`` existed has unknown provenance, so it
    counts as changed; rebuilding it costs only a local decode because the
    source ``.hdr`` is kept.
    """
    if row is None:
        return False
    recorded = row.get("tonemap_key")
    if recorded is None:
        return True
    try:
        return not math.isclose(
            float(recorded), float(key), rel_tol=1e-12, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return True


def _is_cached(out_dir: Path, row: dict[str, Any] | None) -> bool:
    """True when the stored JPG still matches the manifest's ``jpg_md5``."""
    if not row:
        return False
    recorded = row.get("jpg_md5")
    name = row.get("file")
    if not recorded or not name:
        return False
    jpg = Path(out_dir) / name
    if not jpg.is_file():
        return False
    return md5_of_file(jpg) == recorded


def _tonemap_to_jpg(
    hdr_path: Path, jpg_path: Path, jpeg_quality: int, key: float
) -> tuple[int, int]:
    """Decode, tonemap and store one JPG; returns ``(width, height)``."""
    radiance = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if radiance is None:
        raise RuntimeError(f"could not decode {Path(hdr_path).name}")
    image = tonemap_reinhard(radiance, key=key)
    if not cv2.imwrite(
        str(jpg_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ):
        raise RuntimeError(f"could not write {Path(jpg_path).name}")
    height, width = image.shape[:2]
    return width, height


def _entry(
    asset_id: str,
    asset: dict[str, Any],
    hdr_path: Path,
    jpg_path: Path,
    width: int,
    height: int,
    md5: str,
    source_url: str,
    download_format: str,
    key: float,
) -> ManifestEntry:
    return ManifestEntry(
        asset_id=asset_id,
        name=str(asset.get("name", "")),
        categories=list(asset.get("categories") or []),
        category=str(asset.get("category", "")),
        primary_category=primary_category(asset),
        attributes=dict(asset.get("attributes") or {}),
        authors=dict(asset.get("authors") or {}),
        width=int(width),
        height=int(height),
        md5=md5,
        jpg_md5=md5_of_file(jpg_path),
        source_url=source_url,
        download_format=download_format,
        file=Path(jpg_path).name,
        source_file=Path(hdr_path).name,
        tonemap_key=float(key),
    )


def download_asset(
    client: PolyHavenClient,
    asset_id: str,
    asset: dict[str, Any],
    out_dir: Path,
    res: str = DEFAULT_RES,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    key: float = DEFAULT_KEY,
    log: Callable[[str], None] = lambda _msg: None,
) -> ManifestEntry:
    """Fetch (or reuse), verify, tonemap and store one asset."""
    out_dir = Path(out_dir)
    choice = pick_download(client.asset_files(asset_id), res)
    expected_md5 = choice.get("md5")
    hdr_path = source_path(out_dir, asset_id, res, choice["download_format"])

    if hdr_path.is_file() and expected_md5 and md5_of_file(hdr_path) == expected_md5:
        actual_md5 = expected_md5
        log(f"{asset_id}: reusing local {hdr_path.name}")
    else:
        actual_md5 = client.download(choice["url"], hdr_path)
        if expected_md5 and actual_md5 != expected_md5:
            hdr_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{asset_id}: MD5 mismatch, expected {expected_md5} got {actual_md5}"
            )

    jpg_path = out_dir / f"{asset_id}.jpg"
    try:
        width, height = _tonemap_to_jpg(hdr_path, jpg_path, jpeg_quality, key)
    except RuntimeError as exc:
        raise RuntimeError(f"{asset_id}: {exc}") from exc

    return _entry(
        asset_id, asset, hdr_path, jpg_path, width, height,
        actual_md5, choice["url"], choice["download_format"], key,
    )


def retonemap_asset(
    asset_id: str,
    asset: dict[str, Any],
    previous: dict[str, Any] | None,
    out_dir: Path,
    res: str = DEFAULT_RES,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    key: float = DEFAULT_KEY,
) -> ManifestEntry | None:
    """Rebuild the JPG from the local ``.hdr`` without any request.

    Returns ``None`` when there is nothing to work from -- no manifest row, no
    recorded MD5, no local file, or a file whose MD5 no longer matches -- which
    tells the caller to fall back to the normal download path.
    """
    if not previous:
        return None
    recorded = previous.get("md5")
    if not recorded:
        return None

    fmt = str(previous.get("download_format") or DOWNLOAD_FORMAT)
    hdr_path = source_path(out_dir, asset_id, res, fmt)
    if not hdr_path.is_file() or md5_of_file(hdr_path) != recorded:
        return None

    jpg_path = Path(out_dir) / f"{asset_id}.jpg"
    try:
        width, height = _tonemap_to_jpg(hdr_path, jpg_path, jpeg_quality, key)
    except RuntimeError as exc:
        raise RuntimeError(f"{asset_id}: {exc}") from exc

    return _entry(
        asset_id, asset, hdr_path, jpg_path, width, height,
        recorded, str(previous.get("source_url", "")), fmt, key,
    )


# --- driver ----------------------------------------------------------------


def run(
    out_dir: Path,
    manifest_path: Path,
    res: str = DEFAULT_RES,
    limit: int | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    key: float = DEFAULT_KEY,
    retonemap: bool = False,
    client: PolyHavenClient | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Download (or re-tonemap) the selected assets, merging into the manifest.

    The manifest starts from whatever is already on disk and is rewritten after
    every asset, so interrupting the run keeps the work already done and a
    ``--limit`` run never discards rows outside its selection.
    """
    out_dir = Path(out_dir)
    manifest_path = Path(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    client = client if client is not None else PolyHavenClient()
    rows = load_manifest(manifest_path)
    carried = len(rows)

    assets = client.list_hdris()
    asset_ids = select_asset_ids(assets, limit)
    log(
        f"{len(assets)} HDRIs available; processing {len(asset_ids)} at res={res}"
        + (f"; {carried} already in manifest" if carried else "")
    )

    def flush() -> dict[str, Any]:
        manifest = build_manifest(rows, res, client.user_agent, key)
        write_manifest(manifest_path, manifest)
        return manifest

    manifest = flush()
    processed = skipped = failed = 0
    total = len(asset_ids)
    for index, asset_id in enumerate(asset_ids, start=1):
        previous = rows.get(asset_id)
        # A different --key invalidates the stored JPG just as surely as a
        # missing one, so the cache check must not short-circuit it.
        key_changed = _tonemap_key_changed(previous, key)
        rebuild = retonemap or key_changed
        try:
            entry: ManifestEntry | None = None
            if rebuild:
                entry = retonemap_asset(
                    asset_id, assets[asset_id], previous, out_dir, res, jpeg_quality, key
                )
                if entry is not None:
                    reason = "key changed" if key_changed else "local source"
                    log(f"[{index}/{total}] {asset_id}: re-tonemapped ({reason})")
            if entry is None:
                if not rebuild and _is_cached(out_dir, previous):
                    skipped += 1
                    log(f"[{index}/{total}] {asset_id}: cached, skipping")
                    continue
                entry = download_asset(
                    client, asset_id, assets[asset_id], out_dir, res, jpeg_quality,
                    key, log=lambda msg: log(f"[{index}/{total}] {msg}"),
                )
                log(
                    f"[{index}/{total}] {asset_id}: "
                    f"{entry.width}x{entry.height} -> {entry.file}"
                )
        except (LookupError, RuntimeError, OSError, requests.RequestException) as exc:
            failed += 1
            log(f"[{index}/{total}] {asset_id}: FAILED ({exc})")
            continue

        # Persist immediately: a KeyboardInterrupt after this point still leaves
        # every finished asset recorded.
        rows[asset_id] = entry.to_json()
        processed += 1
        manifest = flush()

    log(
        f"wrote {manifest_path} with {len(rows)} assets "
        f"({processed} processed, {skipped} cached, {failed} failed, "
        f"{client.request_count} requests)"
    )
    return manifest


def _default_manifest_path(out_dir: Path) -> Path:
    """``data_root/raw/polyhaven`` -> ``data_root/manifest.json`` (SPEC.md 2.2)."""
    out_dir = Path(out_dir)
    base = out_dir.parents[1] if len(out_dir.parents) >= 2 else out_dir
    return base / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Poly Haven HDRI panoramas (CC0) and write a manifest."
    )
    parser.add_argument("--out", required=True, help="directory for the stored files")
    parser.add_argument(
        "--res", default=DEFAULT_RES, help=f"HDRI resolution (default {DEFAULT_RES})"
    )
    parser.add_argument("--limit", type=int, default=None, help="first N assets only")
    parser.add_argument(
        "--manifest",
        default=None,
        help="manifest path (default <out>/../../manifest.json)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument(
        "--key",
        type=float,
        default=DEFAULT_KEY,
        help=f"Reinhard key / target middle grey (default {DEFAULT_KEY})",
    )
    parser.add_argument(
        "--retonemap",
        action="store_true",
        help="rebuild JPGs from the local .hdr files instead of downloading",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=MIN_REQUEST_INTERVAL,
        help=f"seconds between requests (>= {MIN_REQUEST_INTERVAL})",
    )
    parser.add_argument("--user-agent", default=USER_AGENT)
    return parser


def main(argv: Iterable[str] | None = None, client: PolyHavenClient | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    out_dir = Path(args.out)
    manifest_path = (
        Path(args.manifest) if args.manifest else _default_manifest_path(out_dir)
    )
    if client is None:
        client = PolyHavenClient(
            user_agent=args.user_agent, min_interval=args.min_interval
        )
    run(
        out_dir=out_dir,
        manifest_path=manifest_path,
        res=args.res,
        limit=args.limit,
        jpeg_quality=args.jpeg_quality,
        key=args.key,
        retonemap=args.retonemap,
        client=client,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
