from intersectionqa.geometry.labels import (
    RawGeometry,
    binary_answer,
    derive_labels,
    validate_label_consistency,
    volume_bucket,
)
from intersectionqa.schema import Diagnostics, GeometryLabels, LabelPolicy
import pytest


def test_label_rules_golden_boxes():
    policy = LabelPolicy()
    cases = [
        (
            RawGeometry(1000.0, 1000.0, 0.0, 10.0, aabb_overlap=False),
            "disjoint",
            "no",
            "0",
        ),
        (
            RawGeometry(1000.0, 1000.0, 0.0, 0.0, aabb_overlap=True),
            "touching",
            "no",
            "0",
        ),
        (
            RawGeometry(1000.0, 1000.0, 1.0, 0.0, aabb_overlap=True),
            "intersecting",
            "yes",
            "(0, 0.01]",
        ),
        (
            RawGeometry(1000.0, 216.0, 216.0, 0.0, False, True, True),
            "contained",
            "yes",
            ">0.50",
        ),
        (
            RawGeometry(1000.0, 1000.0, 0.0, 0.5, aabb_overlap=False),
            "near_miss",
            "no",
            "0",
        ),
    ]

    for raw, relation, binary, bucket in cases:
        labels, diagnostics = derive_labels(raw, policy)
        assert labels.relation == relation
        assert binary_answer(labels.relation) == binary
        assert volume_bucket(labels, policy) == bucket
        validate_label_consistency(labels, diagnostics, policy)


def test_volume_bucket_boundaries_are_upper_inclusive():
    policy = LabelPolicy()
    expected = [
        (0.01, "(0, 0.01]"),
        (0.05, "(0.01, 0.05]"),
        (0.20, "(0.05, 0.20]"),
        (0.50, "(0.20, 0.50]"),
        (0.51, ">0.50"),
    ]
    for ratio, bucket in expected:
        labels, _ = derive_labels(
            RawGeometry(1000.0, 1000.0, ratio * 1000.0, 0.0, aabb_overlap=True),
            policy,
        )
        assert volume_bucket(labels, policy) == bucket


def test_label_consistency_rejects_impossible_normalized_overlap():
    policy = LabelPolicy()
    labels = GeometryLabels(
        volume_a=100.0,
        volume_b=10.0,
        intersection_volume=100.0,
        normalized_intersection=10.0,
        minimum_distance=0.0,
        relation="intersecting",
        contained=False,
        contains_a_in_b=False,
        contains_b_in_a=False,
    )
    diagnostics = Diagnostics(
        aabb_overlap=True,
        exact_overlap=True,
        boolean_status="ok",
        distance_status="skipped_positive_overlap",
        label_status="ok",
        failure_reason=None,
    )
    with pytest.raises(ValueError, match="normalized_intersection"):
        validate_label_consistency(labels, diagnostics, policy)


def test_label_consistency_rejects_exact_overlap_with_disjoint_aabbs():
    policy = LabelPolicy()
    labels, diagnostics = derive_labels(
        RawGeometry(100.0, 100.0, 10.0, 0.0, aabb_overlap=False),
        policy,
    )
    with pytest.raises(ValueError, match="disjoint AABBs"):
        validate_label_consistency(labels, diagnostics, policy)


def test_label_consistency_rejects_touching_distance_with_disjoint_aabbs():
    policy = LabelPolicy()
    labels, diagnostics = derive_labels(
        RawGeometry(100.0, 100.0, 0.0, 0.0, aabb_overlap=False),
        policy,
    )
    with pytest.raises(ValueError, match="touching distance"):
        validate_label_consistency(labels, diagnostics, policy)
