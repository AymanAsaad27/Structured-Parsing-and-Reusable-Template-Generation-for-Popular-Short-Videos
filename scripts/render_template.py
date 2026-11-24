"""Render a manual template and explicit asset manifest; no generation/API calls."""

import argparse
import json
from pathlib import Path

from beauty_video_templates.rendering.ffmpeg import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument(
        "--assets", type=Path, required=True, help="JSON mapping run names to clip paths"
    )
    parser.add_argument("--music", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(args.template.read_text(encoding="utf-8"))
    raw = json.loads(args.assets.read_text(encoding="utf-8"))
    clips = {key: (args.assets.parent / Path(value)).resolve() for key, value in raw.items()}
    render(template, clips, args.music, args.output, args.work_dir, args.font)
    print(args.output)


if __name__ == "__main__":
    main()
