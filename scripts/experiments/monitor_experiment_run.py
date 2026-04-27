"""Summarize logs, metrics, disk/GPU state, and stop signals for a run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from intersectionqa.experiments import scan_stop_signals


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tail-lines", type=int, default=20)
    args = parser.parse_args(argv)

    report = {
        "run_dir": str(args.run_dir),
        "status": _read_json(args.run_dir / "status.json"),
        "latest_train_metrics": _latest_jsonl(args.run_dir / "train_metrics.jsonl"),
        "latest_eval_metrics": _latest_jsonl(args.run_dir / "eval_metrics.jsonl"),
        "latest_quality_metrics": _latest_jsonl(args.run_dir / "quality_metrics.jsonl"),
        "disk": _disk(args.run_dir),
        "gpu": _gpu(),
        "stop_signals": scan_stop_signals(args.run_dir),
        "log_tails": {
            str(path.relative_to(args.run_dir)): _tail(path, args.tail_lines)
            for path in sorted((args.run_dir / "logs").glob("*.log"))
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if any(signal.get("severity") == "hard" for signal in report["stop_signals"]):
        raise SystemExit(2)


def _latest_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    latest = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                latest = json.loads(line)
    return latest


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _disk(path: Path) -> dict[str, int | float]:
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_ratio": usage.free / usage.total if usage.total else 0.0,
    }


def _gpu() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return (result.stdout or result.stderr or "").strip()


def _tail(path: Path, line_count: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-line_count:]


if __name__ == "__main__":
    main()

