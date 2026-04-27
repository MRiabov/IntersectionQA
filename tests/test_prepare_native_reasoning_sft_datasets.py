from __future__ import annotations

from scripts.training.prepare_native_reasoning_sft_datasets import (
    long_reasoning_payload,
    manual_audit_template,
    short_reasoning_payload,
    shorten_reasoning,
)


def _payload() -> dict[str, object]:
    return {
        "id": "row-1",
        "task_type": "relation_classification",
        "answer": "intersecting",
        "target_text": (
            "<think>Object A is a box with a flat face. Object B is translated into the box "
            "so the overlap is positive. That means the final relation is intersecting.</think>"
            "<answer>intersecting</answer>"
        ),
        "supervision": {"source_trace_run": "run-1"},
    }


def test_shorten_reasoning_keeps_a_concise_reasoning_window():
    reasoning = (
        "Object A is a box with a flat face. Object B is translated into the box so the overlap is "
        "positive. That means the final relation is intersecting."
    )

    shortened = shorten_reasoning(reasoning, task_type="relation_classification", min_tokens=8, max_tokens=16)

    assert 8 <= len(shortened.split()) <= 16
    assert "intersecting" in shortened


def test_short_and_long_reasoning_payloads_preserve_canonical_answer():
    short_row = short_reasoning_payload(_payload(), min_tokens=8, max_tokens=16)
    long_row = long_reasoning_payload(_payload())

    assert short_row["canonical_answer"] == "intersecting"
    assert short_row["supervision"]["target_text_format"] == "native_reasoning_short_think_answer_v01"
    assert short_row["supervision"]["target_text_source"] == "openrouter_native_reasoning_shortened"
    assert short_row["target_text"].endswith("<answer>intersecting</answer>")

    assert long_row["canonical_answer"] == "intersecting"
    assert long_row["supervision"]["target_text_format"] == "native_reasoning_think_answer_v01"
    assert long_row["supervision"]["target_text_source"] == "openrouter_native_reasoning_long"


def test_manual_audit_template_includes_audit_status_fields():
    template = manual_audit_template(
        [
            {
                "id": "row-1",
                "task_type": "relation_classification",
                "answer": "intersecting",
                "target_text": "<think>reasoning</think><answer>intersecting</answer>",
            }
        ]
    )

    assert "Allowed statuses" in template
    assert "row-1" in template
    assert "- status:" in template
    assert "- note:" in template
