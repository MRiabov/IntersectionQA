"""Render PNG previews for one exported dataset row using PyVista."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir
from intersectionqa.schema import PublicTaskRow
from scripts.dataset.internal.row_artifacts import execute_row_assembly, export_row_artifacts

COLORS = {
    "object_a": "#2f66d0",
    "object_b": "#d77a16",
    "intersection": "#9b0000",
}


def render_row_artifacts(
    dataset_dir: Path,
    row_id: str,
    output_dir: Path,
    *,
    image_size: tuple[int, int] = (1200, 900),
    tessellation_tolerance: float = 0.2,
    ensure_debug_artifacts: bool = True,
) -> dict[str, Any]:
    rows = validate_dataset_dir(dataset_dir)
    matches = [row for row in rows if row.id == row_id]
    if not matches:
        raise ValueError(f"row not found: {row_id}")
    row = matches[0]
    if ensure_debug_artifacts:
        export_row_artifacts(dataset_dir, row.id, output_dir, write_step=True)

    row_dir = output_dir / _safe_row_dir(row.id)
    render_dir = row_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    object_a, object_b = execute_row_assembly(row)
    intersection = None
    if row.diagnostics.exact_overlap:
        candidate = object_a.intersect(object_b)
        if float(candidate.Volume()) > row.label_policy.epsilon_volume(
            row.labels.volume_a or 0.0,
            row.labels.volume_b or 0.0,
        ):
            intersection = candidate

    files: dict[str, str] = {}
    files["object_a"] = str(
        _render_scene(
            render_dir / "object_a.png",
            [(object_a, COLORS["object_a"], 1.0)],
            image_size,
            tessellation_tolerance,
        )
    )
    files["object_b"] = str(
        _render_scene(
            render_dir / "object_b.png",
            [(object_b, COLORS["object_b"], 1.0)],
            image_size,
            tessellation_tolerance,
        )
    )
    assembly_shapes = [(object_a, COLORS["object_a"], 0.72), (object_b, COLORS["object_b"], 0.72)]
    if intersection is not None:
        assembly_shapes.append((intersection, COLORS["intersection"], 1.0))
    files["assembly"] = str(
        _render_scene(
            render_dir / "assembly.png",
            assembly_shapes,
            image_size,
            tessellation_tolerance,
        )
    )
    if intersection is not None:
        files["intersection"] = str(
            _render_scene(
                render_dir / "intersection.png",
                [(intersection, COLORS["intersection"], 1.0)],
                image_size,
                tessellation_tolerance,
            )
        )
    files["contact_sheet"] = str(
        _contact_sheet(
            render_dir / "contact_sheet.png",
            [
                ("object_a", Path(files["object_a"])),
                ("object_b", Path(files["object_b"])),
                ("assembly", Path(files["assembly"])),
                *(
                    [("intersection", Path(files["intersection"]))]
                    if "intersection" in files
                    else []
                ),
            ],
        )
    )

    manifest = {
        "row_id": row.id,
        "task_type": row.task_type,
        "answer": row.answer,
        "relation": row.labels.relation,
        "renderer": "pyvista",
        "image_size": list(image_size),
        "tessellation_tolerance": tessellation_tolerance,
        "files": files,
    }
    manifest_path = render_dir / "render_manifest.json"
    manifest["files"]["render_manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _render_scene(
    path: Path,
    shapes: list[tuple[Any, str, float]],
    image_size: tuple[int, int],
    tessellation_tolerance: float,
) -> Path:
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=image_size)
    plotter.set_background("white")
    for shape, color, opacity in shapes:
        plotter.add_mesh(
            _shape_mesh(shape, tessellation_tolerance),
            color=color,
            opacity=opacity,
            smooth_shading=True,
            specular=0.25,
            specular_power=18,
        )
    plotter.camera_position = "iso"
    plotter.camera.azimuth = 35
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.25)
    plotter.show(screenshot=str(path), auto_close=True)
    return path


def _shape_mesh(shape: Any, tessellation_tolerance: float):
    import pyvista as pv

    vertices, faces = shape.tessellate(tessellation_tolerance)
    points = np.array([[vertex.x, vertex.y, vertex.z] for vertex in vertices], dtype=float)
    face_data: list[int] = []
    for face in faces:
        face_data.extend([len(face), *face])
    return pv.PolyData(points, np.array(face_data, dtype=np.int64))


def _contact_sheet(path: Path, items: list[tuple[str, Path]]) -> Path:
    tile_width = 600
    tile_height = 470
    columns = 2
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), (245, 245, 245))
    for index, (label, image_path) in enumerate(items):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((560, 420), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(image, ((tile_width - image.width) // 2, 30))
        ImageDraw.Draw(tile).text((24, 438), label, fill=(20, 20, 20))
        sheet.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    sheet.save(path, quality=95)
    return path


def _safe_row_dir(row_id: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in row_id).strip("_")


def _image_size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1200x900") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("row_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=_image_size, default=(1200, 900))
    parser.add_argument("--tessellation-tolerance", type=float, default=0.2)
    parser.add_argument(
        "--no-debug-artifacts",
        action="store_true",
        help="skip prompt/JSON/STEP artifact export and write renders only",
    )
    args = parser.parse_args()

    configure_logging()
    manifest = render_row_artifacts(
        args.dataset_dir,
        args.row_id,
        args.output_dir,
        image_size=args.image_size,
        tessellation_tolerance=args.tessellation_tolerance,
        ensure_debug_artifacts=not args.no_debug_artifacts,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
