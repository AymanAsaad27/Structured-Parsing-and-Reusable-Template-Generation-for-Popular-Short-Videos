import pytest

from beauty_video_templates.evaluation.metrics import box_iou, event_f1, word_error_rate


def test_duplicate_detections_cannot_match_twice():
    result = event_f1([9, 10, 11], [10], 2)
    assert result["matched"] == 1
    assert result["precision"] == pytest.approx(1 / 3)


def test_boundary_tolerance_inclusive():
    assert event_f1([8, 22], [10, 20], 2)["f1"] == 1
    assert event_f1([7], [10], 2)["f1"] == 0


def test_empty_event_sets():
    assert event_f1([], [], 2)["f1"] == 1
    assert event_f1([], [1], 2)["f1"] == 0


def test_wer_and_iou():
    assert word_error_rate("a c", "a b") == 0.5
    assert box_iou([0, 0, 1, 1], [0, 0, 0.5, 1]) == 0.5
    with pytest.raises(ValueError):
        word_error_rate("hello", "")
