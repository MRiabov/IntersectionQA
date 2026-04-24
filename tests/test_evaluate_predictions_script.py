import json

from scripts.evaluate_predictions import _read_predictions


def test_read_predictions_jsonl(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"id": "intersectionqa_binary_000001", "output": "yes"}) + "\n",
        encoding="utf-8",
    )
    predictions = _read_predictions(path)
    assert len(predictions) == 1
    assert predictions[0].row_id == "intersectionqa_binary_000001"
    assert predictions[0].output == "yes"
