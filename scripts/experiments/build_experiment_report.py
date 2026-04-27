"""Build a Markdown/JSON summary from saved run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)

    rows = [_summarize_run(path) for path in sorted(args.runs_dir.iterdir()) if path.is_dir()]
    report = {"runs_dir": str(args.runs_dir), "runs": rows}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = _markdown(rows)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


def _summarize_run(run_dir: Path) -> dict:
    manifest = _read_json(run_dir / "run_manifest.json") or {}
    status = _read_json(run_dir / "status.json") or {}
    artifacts = _read_json(run_dir / "artifacts.json") or {"artifacts": []}
    latest_eval = _latest_jsonl(run_dir / "eval_metrics.jsonl") or _latest_jsonl(run_dir / "quality_metrics.jsonl")
    predictions = [
        str(path.relative_to(run_dir))
        for path in sorted((run_dir / "predictions").glob("*.jsonl"))
    ]
    return {
        "run_id": manifest.get("run_id") or run_dir.name,
        "kind": manifest.get("kind"),
        "model": manifest.get("model"),
        "decoding_mode": manifest.get("decoding_mode"),
        "status": status.get("status"),
        "run_dir": str(run_dir),
        "latest_eval": latest_eval,
        "prediction_files": predictions,
        "artifact_count": len(artifacts.get("artifacts", [])),
    }


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    latest = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                latest = json.loads(line)
    return latest


def _markdown(rows: list[dict]) -> str:
    lines = [
        "# Experiment Report",
        "",
        "| Run | Kind | Model | Decoding | Status | Predictions |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {kind} | {model} | {decoding_mode} | {status} | {prediction_count} |".format(
                run_id=row.get("run_id") or "",
                kind=row.get("kind") or "",
                model=row.get("model") or "",
                decoding_mode=row.get("decoding_mode") or "",
                status=row.get("status") or "",
                prediction_count=len(row.get("prediction_files") or []),
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

