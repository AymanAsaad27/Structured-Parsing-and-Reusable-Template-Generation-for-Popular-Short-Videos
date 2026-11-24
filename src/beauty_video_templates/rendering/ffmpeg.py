"""Render a validated, explicit slot timeline using local clips and caption overlays."""

import json
import math
import subprocess
from pathlib import Path


def validate_timeline(template):
    slots = template["slots"]
    if not slots:
        raise ValueError("Timeline must contain slots")
    cursor = 0.0
    ids = set()
    for slot in slots:
        start, end = slot["start"], slot["end"]
        if not all(math.isfinite(v) for v in [start, end]) or end <= start:
            raise ValueError("Slot times must be finite and increasing")
        if abs(start - cursor) > 1e-6:
            raise ValueError("Slots must be contiguous and start at zero")
        if slot["id"] in ids:
            raise ValueError("Slot IDs must be unique")
        ids.add(slot["id"])
        cursor = end
    if abs(cursor - template["duration_seconds"]) > 1e-6:
        raise ValueError("Timeline duration does not equal declared duration")
    return cursor


def render(template, clips, music, output, work_dir, font_path, audio_offset=0):
    import av
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    validate_timeline(template)
    width, height, fps = template["width"], template["height"], template["fps"]
    slots = template["slots"]
    for slot in slots:
        path = Path(clips[slot["run_name"]])
        if not path.is_file():
            raise FileNotFoundError(path)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base)
            if duration + 1 / fps < slot["end"] - slot["start"]:
                raise ValueError(f"Clip too short for {slot['id']}")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    def run(args):
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args],
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-3000:])

    segments = []
    caption_font = ImageFont.truetype(str(font_path), 36)
    small_font = ImageFont.truetype(str(font_path), 18)
    for index, slot in enumerate(slots):
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle((40, 70, width - 40, 225), radius=18, fill=(20, 34, 40, 190))
        words = slot["caption"].replace(" / ", "\n").split("\n")
        lines = []
        for text in words:
            line = ""
            for word in text.split():
                candidate = (line + " " + word).strip()
                if draw.textlength(candidate, font=caption_font) > width - 120 and line:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        for row, line in enumerate(lines[:3]):
            draw.text((64, 92 + row * 44), line, font=caption_font, fill="white")
        draw.rounded_rectangle(
            (40, height - 104, width - 40, height - 49), radius=12, fill=(20, 34, 40, 185)
        )
        draw.text(
            (57, height - 89),
            f"AI-GENERATED DEMO  |  {index + 1:02}/{len(slots):02}",
            font=small_font,
            fill="white",
        )
        overlay_path = work_dir / f"{index:02}_caption.png"
        overlay.save(overlay_path)
        segment = work_dir / f"{index:02}_segment.mp4"
        filters = f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v];[v][1:v]overlay=0:0:shortest=1,format=yuv420p[out]"
        run(
            [
                "-i",
                str(clips[slot["run_name"]]),
                "-loop",
                "1",
                "-i",
                str(overlay_path),
                "-filter_complex",
                filters,
                "-map",
                "[out]",
                "-an",
                "-t",
                str(slot["end"] - slot["start"]),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "medium",
                str(segment),
            ]
        )
        segments.append(segment)
    listing = work_dir / "concat.txt"
    # The renderer controls segment names; source names never become concat directives.
    listing.write_text(
        "\n".join("file '" + p.name + "'" for p in segments) + "\n", encoding="utf-8"
    )
    clean = work_dir / "picture_edit.mp4"
    run(["-f", "concat", "-safe", "1", "-i", str(listing), "-c", "copy", str(clean)])
    duration = template["duration_seconds"]
    run(
        [
            "-i",
            str(clean),
            "-ss",
            str(audio_offset),
            "-i",
            str(music),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-af",
            f"loudnorm=I=-22:TP=-2:LRA=7,afade=t=in:st=0:d=0.25,afade=t=out:st={duration - 0.8}:d=0.8",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-t",
            str(duration),
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    manifest = {
        "template_id": template["template_id"],
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "audio_offset_seconds": audio_offset,
        "caption_method": "separately rasterized overlays",
        "edits": [
            {"slot_id": s["id"], "start": s["start"], "end": s["end"], "asset_run": s["run_name"]}
            for s in slots
        ],
    }
    (output.parent / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output
