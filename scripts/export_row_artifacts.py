"""Export local debug artifacts for one public dataset row."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from intersectionqa.geometry.cadquery_exec import object_to_shape
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir
from intersectionqa.schema import PublicTaskRow


def export_row_artifacts(
    dataset_dir: Path,
    row_id: str,
    output_dir: Path,
    *,
    write_step: bool = True,
) -> dict[str, Any]:
    rows = validate_dataset_dir(dataset_dir)
    matches = [row for row in rows if row.id == row_id]
    if not matches:
        raise ValueError(f"row not found: {row_id}")
    row = matches[0]

    row_dir = output_dir / _safe_name(row.id)
    row_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Any] = {
        "row_id": row.id,
        "task_type": row.task_type,
        "answer": row.answer,
        "relation": row.labels.relation,
        "files": {},
        "step_export_status": "skipped" if not write_step else "not_run",
        "step_export_errors": [],
    }
    _write_text(row_dir / "prompt.txt", row.prompt, artifacts)
    _write_text(row_dir / "answer.txt", row.answer + "\n", artifacts)
    _write_text(row_dir / "assembly.py", row.script, artifacts)
    _write_json(row_dir / "row.json", row.model_dump(mode="json"), artifacts)
    _write_json(
        row_dir / "labels_and_diagnostics.json",
        {
            "labels": row.labels.model_dump(mode="json"),
            "diagnostics": row.diagnostics.model_dump(mode="json"),
            "difficulty_tags": row.difficulty_tags,
            "label_policy": row.label_policy.model_dump(mode="json"),
            "metadata": row.metadata,
        },
        artifacts,
    )

    if write_step:
        _export_step_artifacts(row, row_dir, artifacts)

    manifest_path = row_dir / "artifacts_manifest.json"
    artifacts["files"]["artifacts_manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifacts


def _export_step_artifacts(
    row: PublicTaskRow,
    row_dir: Path,
    artifacts: dict[str, Any],
) -> None:
    import cadquery as cq

    try:
        object_a, object_b = execute_row_assembly(row)
        assembly = cq.Compound.makeCompound([object_a, object_b])
        _export_step(object_a, row_dir / "object_a.step", artifacts, "object_a_step")
        _export_step(object_b, row_dir / "object_b.step", artifacts, "object_b_step")
        _export_step(assembly, row_dir / "assembly.step", artifacts, "assembly_step")
        if row.diagnostics.exact_overlap:
            try:
                intersection = object_a.intersect(object_b)
                if float(intersection.Volume()) > row.label_policy.epsilon_volume(
                    row.labels.volume_a or 0.0,
                    row.labels.volume_b or 0.0,
                ):
                    _export_step(
                        intersection,
                        row_dir / "intersection.step",
                        artifacts,
                        "intersection_step",
                    )
                else:
                    artifacts["step_export_errors"].append(
                        "intersection boolean produced no policy-positive volume"
                    )
            except Exception as exc:  # pragma: no cover - depends on OCC failure mode
                artifacts["step_export_errors"].append(f"intersection_step: {exc}")
        artifacts["step_export_status"] = (
            "ok" if not artifacts["step_export_errors"] else "partial"
        )
    except Exception as exc:
        artifacts["step_export_status"] = "failed"
        artifacts["step_export_errors"].append(str(exc))


def execute_row_assembly(row: PublicTaskRow) -> tuple[Any, Any]:
    import cadquery as cq

    namespace: dict[str, Any] = {"cq": cq, "cadquery": cq, "__builtins__": __builtins__}
    compiled = compile(row.script, f"<intersectionqa-row:{row.id}>", "exec")
    exec(compiled, namespace)
    assembly = namespace.get("assembly")
    if not callable(assembly):
        raise RuntimeError("row script does not define callable assembly()")
    result = assembly()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("assembly() must return a two-item tuple")
    return object_to_shape(result[0]), object_to_shape(result[1])


def _export_step(shape: Any, path: Path, artifacts: dict[str, Any], key: str) -> None:
    import cadquery as cq

    cq.exporters.export(shape, str(path))
    artifacts["files"][key] = str(path)


def _write_text(path: Path, text: str, artifacts: dict[str, Any]) -> None:
    path.write_text(text, encoding="utf-8")
    artifacts["files"][path.stem] = str(path)


def _write_json(path: Path, value: object, artifacts: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts["files"][path.stem] = str(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("row_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-step", action="store_true", help="write text/JSON artifacts only")
    args = parser.parse_args()

    configure_logging()
    manifest = export_row_artifacts(
        args.dataset_dir,
        args.row_id,
        args.output_dir,
        write_step=not args.no_step,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
