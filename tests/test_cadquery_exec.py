import pytest

from intersectionqa.enums import BooleanStatus, DistanceStatus
from intersectionqa.geometry.cadquery_exec import execute_source_object, measure_shape_pair, measure_source_pair
from intersectionqa.schema import LabelPolicy, Transform
from intersectionqa.sources.synthetic import synthetic_source_object


def test_execute_source_object_measures_cadquery_shape():
    source = synthetic_source_object("obj_test", "object_a", (2.0, 3.0, 4.0))
    measured = execute_source_object(source)
    assert measured.volume == pytest.approx(24.0)
    assert measured.bbox.min == pytest.approx((-1.0, -1.5, -2.0))
    assert measured.bbox.max == pytest.approx((1.0, 1.5, 2.0))


def test_measure_source_pair_uses_exact_cadquery_boolean_and_distance():
    policy = LabelPolicy()
    object_a = synthetic_source_object("obj_a", "object_a", (10.0, 10.0, 10.0))
    object_b = synthetic_source_object("obj_b", "object_b", (10.0, 10.0, 10.0))

    tiny_overlap = measure_source_pair(
        object_a,
        object_b,
        Transform(translation=(0.0, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0)),
        Transform(translation=(9.99, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0)),
        policy,
    )
    assert tiny_overlap.intersection_volume == pytest.approx(1.0)
    assert tiny_overlap.minimum_distance == 0.0
    assert tiny_overlap.boolean_status == BooleanStatus.OK
    assert tiny_overlap.distance_status == DistanceStatus.SKIPPED_POSITIVE_OVERLAP

    near_miss = measure_source_pair(
        object_a,
        object_b,
        Transform(translation=(0.0, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0)),
        Transform(translation=(10.5, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0)),
        policy,
    )
    assert near_miss.intersection_volume == 0.0
    assert near_miss.minimum_distance == pytest.approx(0.5)
    assert near_miss.boolean_status == BooleanStatus.SKIPPED_AABB_DISJOINT
    assert near_miss.distance_status == DistanceStatus.OK


def test_measure_shape_pair_does_not_mutate_cached_shapes():
    policy = LabelPolicy()
    object_a = synthetic_source_object("obj_a", "object_a", (10.0, 10.0, 10.0))
    object_b = synthetic_source_object("obj_b", "object_b", (10.0, 10.0, 10.0))
    shape_a = execute_source_object(object_a).shape
    shape_b = execute_source_object(object_b).shape

    first = measure_shape_pair(
        shape_a,
        shape_b,
        Transform(translation=(0.0, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0)),
        Transform(translation=(20.0, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0)),
        policy,
    )
    second = measure_shape_pair(
        shape_a,
        shape_b,
        Transform(translation=(0.0, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0)),
        Transform(translation=(9.99, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0)),
        policy,
    )
    repeated_first = measure_shape_pair(
        shape_a,
        shape_b,
        Transform(translation=(0.0, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0)),
        Transform(translation=(20.0, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0)),
        policy,
    )

    assert first.intersection_volume == pytest.approx(0.0)
    assert second.intersection_volume == pytest.approx(1.0)
    assert repeated_first.intersection_volume == pytest.approx(first.intersection_volume)
    assert repeated_first.minimum_distance == pytest.approx(first.minimum_distance)
