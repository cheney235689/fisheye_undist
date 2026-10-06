"""Run a list of training configs in sequence (SPEC.md 3.13).

    uv run python sweep.py configs/sweeps/<name>.txt

The list file holds one ``configs/train/*.yaml`` path per line; blank lines and
``#`` comments are ignored.

Two properties make a sweep usable over several sessions:

**Resumable.**  A config whose ``run_name`` already has a FINISHED run in the
MLflow store is skipped.  So a sweep interrupted after two of five arms picks
up at the third, and adding an arm to the list re-runs only that arm.

**Fault-tolerant.**  One arm raising -- a bad model name, an OOM, a missing
split -- is reported and the sweep moves on.  Losing the remaining arms because
the second one failed would be the worst possible behaviour for something left
running overnight.

Training is called in-process rather than through ``subprocess``, so a failure
arrives as an exception with a real traceback.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from train import DEFAULT_TRACKING_URI, TrainConfig, load_config, train

__all__ = [
    "EXPERIMENT_NAME",
    "SweepResult",
    "read_config_list",
    "finished_run_exists",
    "run_sweep",
    "main",
]

# Must match the experiment train.py logs into.
EXPERIMENT_NAME = "fisheye-calib"


@dataclass
class SweepResult:
    """Outcome of one arm."""

    config_path: str
    run_name: str
    status: str  # done | skipped | failed
    best_val_reproj_mean_px: float | None = None
    error: str | None = None
    elapsed_sec: float = 0.0


@dataclass
class SweepSummary:
    results: list[SweepResult] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)

    @property
    def done(self) -> int:
        return self.count("done")

    @property
    def skipped(self) -> int:
        return self.count("skipped")

    @property
    def failed(self) -> int:
        return self.count("failed")


def read_config_list(path: str | Path) -> list[Path]:
    """One config path per line; blank lines and ``#`` comments dropped."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")

    configs: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            configs.append(Path(line))
    if not configs:
        raise ValueError(f"{path} lists no configs")
    return configs


def finished_run_exists(
    run_name: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
) -> bool:
    """True when this ``run_name`` already has a FINISHED run (SPEC.md 3.13)."""
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return False

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"attributes.run_name = '{run_name}' "
            "and attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    return bool(runs)


def run_sweep(
    list_path: str | Path,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    out_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
    train_fn: Callable[..., dict[str, Any]] = train,
    experiment_name: str = EXPERIMENT_NAME,
) -> SweepSummary:
    """Run every arm in order, skipping what is done and surviving what fails."""
    configs = read_config_list(list_path)
    summary = SweepSummary()

    log(f"sweep {Path(list_path)}: {len(configs)} configs")
    for position, config_path in enumerate(configs, start=1):
        head = f"[{position}/{len(configs)}] {config_path}"

        # A broken yaml is an arm failure like any other, not a sweep failure.
        try:
            cfg: TrainConfig = load_config(config_path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            log(f"{head}: FAILED to load ({exc})")
            summary.results.append(
                SweepResult(str(config_path), "?", "failed", error=str(exc))
            )
            continue

        if finished_run_exists(cfg.run_name, tracking_uri, experiment_name):
            log(f"{head}: {cfg.run_name} already FINISHED, skipping")
            summary.results.append(
                SweepResult(str(config_path), cfg.run_name, "skipped")
            )
            continue

        log(f"{head}: training {cfg.run_name}")
        started = time.time()
        try:
            result = train_fn(
                cfg, tracking_uri=tracking_uri, out_dir=out_dir, log=log
            )
        except Exception as exc:  # noqa: BLE001 - one bad arm must not end the sweep
            log(f"{head}: FAILED after {time.time() - started:.1f}s ({exc})")
            log(traceback.format_exc())
            summary.results.append(
                SweepResult(
                    str(config_path), cfg.run_name, "failed",
                    error=str(exc), elapsed_sec=time.time() - started,
                )
            )
            continue

        summary.results.append(
            SweepResult(
                str(config_path), cfg.run_name, "done",
                best_val_reproj_mean_px=result.get("best_val_reproj_mean_px"),
                elapsed_sec=time.time() - started,
            )
        )

    _print_summary(summary, log)
    return summary


def _print_summary(summary: SweepSummary, log: Callable[[str], None]) -> None:
    log("")
    log(
        f"sweep summary: {summary.done} done, {summary.skipped} skipped, "
        f"{summary.failed} failed"
    )
    if not summary.results:
        return

    width = max(len(r.run_name) for r in summary.results)
    log(f"  {'run_name':{width}s}  {'status':8s}  {'best val reproj':>15s}")
    for result in summary.results:
        if result.best_val_reproj_mean_px is None:
            score = "-" if result.status != "failed" else "(failed)"
        else:
            score = f"{result.best_val_reproj_mean_px:.4f} px"
        log(f"  {result.run_name:{width}s}  {result.status:8s}  {score:>15s}")
        if result.error:
            log(f"  {'':{width}s}  -> {result.error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a list of training configs in sequence (SPEC.md 3.13)."
    )
    parser.add_argument("list", help="configs/sweeps/<name>.txt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    summary = run_sweep(args.list)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
