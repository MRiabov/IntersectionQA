from intersectionqa.training.prompt_features import augment_prompt


def test_prompt_feature_mode_none_preserves_prompt():
    assert (
        augment_prompt(
            prompt="Base prompt",
            task_type="repair_translation",
            metadata={"bbox_a": {"min": [0, 0, 0], "max": [1, 1, 1]}},
            mode="none",
        )
        == "Base prompt"
    )


def test_edit_geometry_repair_features_expose_aabbs_without_direct_answer():
    prompt = augment_prompt(
        prompt="Repair prompt",
        task_type="repair_translation",
        metadata={
            "bbox_a": {"min": [-1.0, -2.0, -3.0], "max": [1.0, 2.0, 3.0]},
            "bbox_b": {"min": [0.5, -1.0, -1.0], "max": [2.5, 1.0, 1.0]},
            "label_policy": {"epsilon_distance_mm": 0.0001},
            "selected_direction": "+x",
            "selected_magnitude_mm": 1.5001,
            "candidate_moves": [{"direction": "+x", "magnitude_mm": 1.5001}],
        },
        mode="edit_geometry",
    )

    assert "Trusted geometry features" in prompt
    assert "object_a_world_aabb: min=(-1, -2, -3), max=(1, 2, 3)" in prompt
    assert "object_b_world_aabb: min=(0.5, -1, -1), max=(2.5, 1, 1)" in prompt
    assert "contact_tolerance_mm: 0.0001" in prompt
    assert "selected_direction" not in prompt
    assert "selected_magnitude" not in prompt
    assert "candidate_moves" not in prompt


def test_candidate_feature_mode_exposes_ordered_repair_candidates_without_selected_label():
    prompt = augment_prompt(
        prompt="Repair prompt",
        task_type="repair_translation",
        metadata={
            "bbox_a": {"min": [-1.0, -2.0, -3.0], "max": [1.0, 2.0, 3.0]},
            "bbox_b": {"min": [0.5, -1.0, -1.0], "max": [2.5, 1.0, 1.0]},
            "label_policy": {"epsilon_distance_mm": 0.0001},
            "selected_direction": "+x",
            "selected_magnitude_mm": 1.5001,
            "candidate_moves": [
                {"direction": "+z", "magnitude_mm": 4.0},
                {"direction": "-x", "magnitude_mm": 3.0},
                {"direction": "+x", "magnitude_mm": 1.5001},
                {"direction": "-z", "magnitude_mm": 6.0},
                {"direction": "+y", "magnitude_mm": 2.0},
                {"direction": "-y", "magnitude_mm": 5.0},
            ],
        },
        mode="edit_geometry_with_candidates",
    )

    assert (
        "conservative_axis_move_options_mm: "
        "+x=1.500100, -x=3.000000, +y=2.000000, -y=5.000000, "
        "+z=4.000000, -z=6.000000"
    ) in prompt
    assert "selected_direction" not in prompt
    assert "selected_magnitude" not in prompt
    assert "candidate_moves" not in prompt


def test_edit_geometry_signed_distance_features_expose_initial_and_target_state():
    prompt = augment_prompt(
        prompt="Move prompt",
        task_type="target_clearance_move",
        metadata={
            "allowed_edit": {
                "direction": "-x",
                "positive_distance_effect": "moves object_b farther from object_a",
                "negative_distance_effect": "moves object_b closer to object_a",
            },
            "initial_state": {"relation": "disjoint", "minimum_distance": 50.563891},
            "target": {"target_clearance_mm": 1.0},
            "selected_signed_distance_mm": -49.6,
        },
        mode="edit_geometry",
    )

    assert "allowed_signed_direction: -x" in prompt
    assert "initial_clearance_mm: 50.563891" in prompt
    assert "target_clearance_mm: 1" in prompt
    assert "selected_signed_distance" not in prompt


def test_edit_geometry_ignores_non_edit_tasks():
    assert (
        augment_prompt(
            prompt="QA prompt",
            task_type="binary_interference",
            metadata={"bbox_a": {"min": [0, 0, 0], "max": [1, 1, 1]}},
            mode="edit_geometry",
        )
        == "QA prompt"
    )
