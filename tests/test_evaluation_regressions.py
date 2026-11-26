import pytest

from beauty_video_templates.evaluation.metrics import classification_report, match_boxes
from beauty_video_templates.evaluation.runner import evaluate


def test_fixed_taxonomy_counts_absent_class():
    result = classification_report([0, 0], [0, 0], [0, 1])
    assert result["accuracy"] == 1
    assert result["macro_f1"] == 0.5
    assert result["per_class"][1]["support"] == 0


def test_ocr_different_frames_cannot_match():
    p = [{"start": 0, "sample_time": 0, "bbox": [0, 0, 1, 1]}]
    r = [{"start": 1, "sample_time": 1, "bbox": [0, 0, 1, 1]}]
    assert match_boxes(p, r) == []


def test_ocr_matching_can_reassign_for_maximum_cardinality():
    p = [{"start": 0, "bbox": [0, 0, 1, 1]}, {"start": 0, "bbox": [0, 0, 0.5, 1]}]
    r = [{"start": 0, "bbox": [0, 0, 0.5, 1]}, {"start": 0, "bbox": [0.5, 0, 1, 1]}]
    assert len(match_boxes(p, r)) == 2


def test_shot_frame_timebase_ignores_bad_seconds():
    pred = {
        "module": "shots",
        "video": {"id": "v"},
        "items": [{"start": 0, "start_frame": 0}, {"start": 18, "start_frame": 873}],
    }
    ref = {
        "module": "shots",
        "video": {"id": "v"},
        "items": [{"start": 0, "start_frame": 0}, {"start": 14.55, "start_frame": 873}],
    }
    assert evaluate(pred, ref, tolerance=2, timebase="frames")["f1"] == 1


def test_empty_speech_reference_not_reported_perfect():
    p = {"module": "asr", "video": {"id": "v"}, "items": [{"text": "imagined speech"}]}
    r = {"module": "asr", "video": {"id": "v"}, "items": []}
    score = evaluate(p, r)
    assert score["wer"] is None and score["hypothesis_words"] == 2


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        classification_report([0], [2], [0, 1])


def test_ocr_missed_boxes_count_in_text_and_reference_iou():
    p = {
        "module": "ocr",
        "video": {"id": "v"},
        "items": [{"start": 0, "bbox": [0, 0, 1, 1], "text": "one"}],
    }
    r = {
        "module": "ocr",
        "video": {"id": "v"},
        "items": [
            {"start": 0, "bbox": [0, 0, 1, 1], "text": "one"},
            {"start": 1, "bbox": [0, 0, 1, 1], "text": "two"},
        ],
    }
    score = evaluate(p, r)
    assert score["mean_matched_iou"] == 1
    assert score["mean_reference_iou_including_misses"] == 0.5
    assert score["text_accuracy_one_minus_cer_clipped"] == 0.5
