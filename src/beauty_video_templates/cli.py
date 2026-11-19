"""Lightweight commands; no model downloads or inference."""

import argparse
import json
import shutil
import sys
from pathlib import Path

from jsonschema.exceptions import ValidationError

from beauty_video_templates.contracts import validate_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Beauty Video Templates toolkit")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report Python and media executable availability")
    validate = commands.add_parser("validate", help="Validate a parsing result JSON")
    validate.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "python": sys.version.split()[0],
                    "ffmpeg": shutil.which("ffmpeg"),
                    "ffprobe": shutil.which("ffprobe"),
                    "note": "Model/GPU inference is not tested by this command.",
                },
                indent=2,
            )
        )
        return 0
    try:
        validate_result(json.loads(args.path.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Invalid result: {exc}", file=sys.stderr)
        return 1
    print(f"Valid module result: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
