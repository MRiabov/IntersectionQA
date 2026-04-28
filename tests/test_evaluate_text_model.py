from __future__ import annotations

from collections import Counter

from scripts.training.evaluate_text_model import summarize_format


def test_summarize_format_reports_invalid_and_tag_rates() -> None:
    summary = summarize_format(
        Counter(
            {
                "total": 5,
                "parse_valid": 4,
                "invalid": 1,
                "parsed_correct": 3,
                "answer_tag": 2,
                "reasoning_format": 1,
            }
        )
    )

    assert summary["parse_valid_rate"] == 0.8
    assert summary["invalid_rate"] == 0.2
    assert summary["parsed_accuracy"] == 0.6
    assert summary["answer_tag_rate"] == 0.4
    assert summary["reasoning_format_rate"] == 0.2
