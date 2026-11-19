import json
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from beauty_video_templates.cli import main
from beauty_video_templates.contracts import validate_result


@pytest.fixture
def result():
    return json.loads((Path(__file__).parents[1] / "examples/results/shots.json").read_text())


def test_valid_and_empty(result):
    validate_result(result)
    result["items"] = []
    validate_result(result)


@pytest.mark.parametrize(
    "item",
    [
        {"start": 2, "end": 1},
        {"start": 0, "end": 6},
        {"start": -1, "end": 1},
        {"time": float("nan")},
        {"start": 0, "end": 1, "time": 0.5},
        {"time": 1, "start": 0},
        {"start": 1, "end": 1},
        {"time": float("inf")},
    ],
)
def test_invalid_timestamps(result, item):
    result["items"] = [item]
    with pytest.raises((ValueError, ValidationError)):
        validate_result(result)


def test_beat_timestamp(result):
    result["module"] = "beats"
    result["items"] = [{"time": 1.25}]
    validate_result(result)


def test_cli_invalid_json(tmp_path, capsys):
    target = tmp_path / "bad.json"
    target.write_text("{}")
    assert main(["validate", str(target)]) == 1
    assert "Invalid result" in capsys.readouterr().err


def test_cli_missing_file(tmp_path):
    assert main(["validate", str(tmp_path / "missing.json")]) == 1


def test_schema_version(result):
    result["schema_version"] = "99"
    with pytest.raises(ValidationError):
        validate_result(result)
