"""Record installed versions without collecting credentials."""

import json
import platform
import shutil
from importlib.metadata import distributions

print(
    json.dumps(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
            "packages": sorted(
                [{"name": d.metadata["Name"], "version": d.version} for d in distributions()],
                key=lambda item: item["name"].lower(),
            ),
        },
        indent=2,
    )
)
