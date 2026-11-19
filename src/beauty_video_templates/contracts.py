"""Shared parsing contracts without model dependencies."""

import json
import math
from abc import ABC, abstractmethod
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator


def validate_result(result: dict) -> None:
    """Check the common envelope and original-timeline timestamp invariants."""
    schema = json.loads(
        files("beauty_video_templates").joinpath("schemas/module-result.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(result)
    duration = result["video"]["duration_seconds"]
    if not math.isfinite(duration):
        raise ValueError("Video duration must be finite")
    for item in result["items"]:
        keys = ("time",) if "time" in item else ("start", "end")
        if any(not math.isfinite(item[key]) or item[key] > duration for key in keys):
            raise ValueError("Timestamps must be finite and within video duration")
        if "start" in item and item["end"] <= item["start"]:
            raise ValueError("Intervals must have positive duration")


class ParsingBackend(ABC):
    """Concrete Phase 2 backends implement this interface."""

    @abstractmethod
    def parse(self, video_path: Path) -> dict:
        """Analyze real media and return a validated module result."""
        raise NotImplementedError
