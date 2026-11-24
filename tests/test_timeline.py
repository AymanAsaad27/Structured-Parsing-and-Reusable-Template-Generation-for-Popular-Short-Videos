import pytest

from beauty_video_templates.rendering.ffmpeg import validate_timeline


def test_declared_duration_and_gap():
    template = {
        "duration_seconds": 8,
        "slots": [{"id": "a", "start": 0, "end": 4}, {"id": "b", "start": 4, "end": 8}],
    }
    assert validate_timeline(template) == 8
    template["slots"][1]["start"] = 5
    with pytest.raises(ValueError):
        validate_timeline(template)


def test_duplicate_and_nonfinite_slots():
    with pytest.raises(ValueError):
        validate_timeline(
            {
                "duration_seconds": 8,
                "slots": [{"id": "a", "start": 0, "end": 4}, {"id": "a", "start": 4, "end": 8}],
            }
        )
    with pytest.raises(ValueError):
        validate_timeline(
            {"duration_seconds": 4, "slots": [{"id": "a", "start": 0, "end": float("nan")}]}
        )
