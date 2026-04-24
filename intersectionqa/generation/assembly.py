"""Two-object assembly script and representation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from intersectionqa.geometry.transforms import format_transform_block
from intersectionqa.schema import SourceObjectRecord, Transform


@dataclass(frozen=True)
class TwoObjectAssembly:
    object_a: SourceObjectRecord
    object_b: SourceObjectRecord
    transform_a: Transform
    transform_b: Transform

    @property
    def object_code(self) -> str:
        return normalize_pair_code(self.object_a.normalized_code, self.object_b.normalized_code)

    @property
    def transform_block(self) -> str:
        return format_transform_block(self.transform_a, self.transform_b)

    def script(self) -> str:
        return build_assembly_script(self.object_code, self.transform_a, self.transform_b)


def normalize_pair_code(code_a: str, code_b: str) -> str:
    code = "\n".join(
        [
            "import cadquery as cq",
            "",
            code_a.strip(),
            "",
            code_b.strip(),
            "",
        ]
    )
    return code


def build_assembly_script(object_code: str, transform_a: Transform, transform_b: Transform) -> str:
    return "\n".join(
        [
            object_code.strip(),
            "",
            "def _place(solid, translation, rotation_xyz_deg):",
            "    return (",
            "        solid",
            "        .rotate((0, 0, 0), (1, 0, 0), rotation_xyz_deg[0])",
            "        .rotate((0, 0, 0), (0, 1, 0), rotation_xyz_deg[1])",
            "        .rotate((0, 0, 0), (0, 0, 1), rotation_xyz_deg[2])",
            "        .translate(tuple(translation))",
            "    )",
            "",
            "def assembly():",
            f"    a = _place(object_a(), {tuple(transform_a.translation)!r}, {tuple(transform_a.rotation_xyz_deg)!r})",
            f"    b = _place(object_b(), {tuple(transform_b.translation)!r}, {tuple(transform_b.rotation_xyz_deg)!r})",
            "    return a, b",
            "",
        ]
    )
