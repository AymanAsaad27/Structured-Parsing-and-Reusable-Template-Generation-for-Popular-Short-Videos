import pytest

from beauty_video_templates.evaluation.runner import evaluate


def result(module, items):
    return {"module": module, "video": {"id": "sample"}, "items": items}


def test_label_matching_is_one_to_one():
    p = result(
        "transitions", [{"start": 1, "label": "hard_cut"}, {"start": 1.01, "label": "hard_cut"}]
    )
    r = result("transitions", [{"start": 1, "label": "hard_cut"}])
    scores = evaluate(p, r)
    assert scores["matched"] == 1
    assert scores["unmatched_predictions"] == 1


def test_ocr_counts_missing_and_hallucinated_text():
    p = result("ocr", [{"start": 0, "end": 1, "bbox": [0, 0, 1, 1], "text": "cat"}])
    r = result("ocr", [{"start": 0, "end": 1, "bbox": [0, 0, 1, 1], "text": "car"}])
    scores = evaluate(p, r)
    assert scores["mean_matched_iou"] == 1
    assert scores["character_error_rate_including_unmatched"] == pytest.approx(1 / 3)


def test_reference_identity_must_match():
    with pytest.raises(ValueError):
        evaluate(result("asr", []), result("beats", []))
