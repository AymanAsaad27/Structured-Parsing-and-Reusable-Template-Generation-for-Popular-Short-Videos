"""asr parsing module."""

from pathlib import Path

from beauty_video_templates.contracts import ParsingBackend


class Backend(ParsingBackend):
    """Configured adapter with a shared parse(video_path) interface."""

    def __init__(self, **options):
        self.options = options

    def parse(self, video_path: Path) -> dict:
        from .backends import asr

        return asr(video_path, **self.options)
