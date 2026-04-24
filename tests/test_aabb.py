from intersectionqa.config import DatasetConfig
from intersectionqa.evaluation.aabb import evaluate_aabb_binary
from intersectionqa.evaluation.obb import OrientedBox, evaluate_obb_binary, obb_overlap
from intersectionqa.pipeline import build_smoke_rows


def test_aabb_baseline_runs_on_binary_rows():
    rows, _ = build_smoke_rows(DatasetConfig())
    result = evaluate_aabb_binary(rows)
    assert result.total > 0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.per_relation_accuracy is not None
    assert "disjoint" in result.per_relation_accuracy


def test_obb_overlap_detects_oriented_separation_inside_overlapping_aabbs():
    left = OrientedBox(
        center=(0.0, 0.0, 0.0),
        axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        half_extents=(5.0, 0.5, 1.0),
    )
    diagonal = 2**0.5 / 2.0
    right = OrientedBox(
        center=(7.0, 0.0, 0.0),
        axes=((diagonal, diagonal, 0.0), (-diagonal, diagonal, 0.0), (0.0, 0.0, 1.0)),
        half_extents=(5.0, 0.5, 1.0),
    )

    assert not obb_overlap(left, right)


def test_obb_baseline_runs_on_binary_rows():
    rows, _ = build_smoke_rows(DatasetConfig())
    result = evaluate_obb_binary(rows)
    assert result.total > 0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.invalid_output_rate == 0.0
    assert result.per_difficulty_accuracy is not None
    assert "rotated" in result.per_difficulty_accuracy
