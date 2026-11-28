"""Instantiate a reusable demo template using only the supplied new materials."""

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from beauty_video_templates.evaluation.metrics import event_f1
from beauty_video_templates.parsing.media import metadata
from beauty_video_templates.templates.demo_template import validate_template


def find_font():
    for path in [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]:
        if path.is_file():
            return path
    raise FileNotFoundError("Set up a local TrueType font before rendering.")


def run_ffmpeg(args):
    import imageio_ffmpeg

    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *map(str, args),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode:
        raise RuntimeError("Video rendering failed: " + result.stderr[-1800:])


def draw_caption(path, text, box, width, height):
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if text.strip():
        draw = ImageDraw.Draw(overlay)
        left, top, right, bottom = [
            int(v * scale) for v, scale in zip(box, [width, height, width, height])
        ]
        pad = max(8, round(width * 0.025))
        available = right - left - 2 * pad
        size = max(12, round(width * 0.045))
        while True:
            font = ImageFont.truetype(str(find_font()), size)
            lines = []
            for paragraph in text.split("\n"):
                line = ""
                for word in paragraph.split():
                    candidate = (line + " " + word).strip()
                    if draw.textlength(candidate, font=font) <= available:
                        line = candidate
                    else:
                        if line:
                            lines.append(line)
                        line = ""
                        # Long unspaced tokens (including CJK) can wrap by character.
                        for char in word:
                            if line and draw.textlength(line + char, font=font) > available:
                                lines.append(line)
                                line = ""
                            line += char
                if line:
                    lines.append(line)
            if len(lines) <= 3 or size <= 12:
                break
            size -= 2
        line_height = round(size * 1.35)
        actual_height = max(bottom - top, line_height * len(lines) + 2 * pad)
        top = max(0, min(top, height - actual_height - 8))
        draw.rounded_rectangle(
            (left, top, right, top + actual_height),
            radius=max(8, width // 60),
            fill=(16, 21, 29, 205),
        )
        for row, line in enumerate(lines):
            x = left + (right - left - draw.textlength(line, font=font)) / 2
            draw.text((x, top + pad + row * line_height), line, font=font, fill="white")
    overlay.save(path)


def render_demo(
    template,
    assignments,
    assets,
    output_dir,
    *,
    music=None,
    captions=True,
    reference=None,
    progress=None,
):
    validate_template(template)
    if len(assignments) != len(template["slots"]) or any(
        s["id"] != a["slot_id"] for s, a in zip(template["slots"], assignments)
    ):
        raise ValueError("Every template slot needs exactly one matching material assignment.")
    progress = progress or (lambda value, message: None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height, fps = template["width"], template["height"], template["fps"]
    indexed = {a["id"]: a for a in assets}
    palette = template.get("style", {}).get("palette", ["#ece7e2"])
    background = palette[0].replace("#", "0x")
    edits = []
    segments = []
    for i, (slot, choice) in enumerate(zip(template["slots"], assignments)):
        asset = indexed[choice["asset_id"]]
        path = Path(asset["path"])
        if not path.is_file():
            raise FileNotFoundError("A selected product material is missing.")
        count = round(slot["end"] * fps) - round(slot["start"] * fps)
        if count < 1:
            raise ValueError("A slot is shorter than one export frame.")
        duration = count / fps
        input_args = ["-loop", "1", "-i", path] if asset["kind"] == "image" else ["-i", path]
        speed = 1.0
        offset = 0
        if asset["kind"] == "video":
            source_duration = asset["duration_seconds"]
            if source_duration < duration:
                speed = min(4, duration / source_duration)
            elif source_duration > duration:
                offset = min(max(0, source_duration - duration), choice["reuse_index"] * 0.35)
                input_args = ["-ss", round(offset, 4), "-i", path]
        overlay = output_dir / f"caption_{i:03}.png"
        draw_caption(
            overlay,
            slot.get("caption", "") if captions else "",
            slot.get("caption_box", [0.07, 0.73, 0.93, 0.87]),
            width,
            height,
        )
        segment = output_dir / f"segment_{i:03}.mp4"
        filters = f"[0:v]setpts=(PTS-STARTPTS)*{speed},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={background},setsar=1,fps={fps},tpad=stop_mode=clone:stop_duration={duration}[base];[base][1:v]overlay=0:0:shortest=1,format=yuv420p[out]"
        run_ffmpeg(
            [
                *input_args,
                "-loop",
                "1",
                "-i",
                overlay,
                "-filter_complex",
                filters,
                "-map",
                "[out]",
                "-an",
                "-frames:v",
                count,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-threads",
                "2",
                segment,
            ]
        )
        segments.append(segment)
        edits.append(
            {
                **choice,
                "start": round(slot["start"] * fps) / fps,
                "end": round(slot["end"] * fps) / fps,
                "frames": count,
                "playback_rate": round(1 / speed, 4),
                "source_offset": offset,
                "looped": False,
                "frozen_extension_seconds": round(
                    max(0, duration - speed * asset["duration_seconds"]), 4
                )
                if asset["kind"] == "video"
                else 0,
            }
        )
        progress(
            10 + int(65 * (i + 1) / len(assignments)),
            f"Rendering shot {i + 1} of {len(assignments)}",
        )
    listing = output_dir / "concat.txt"
    listing.write_text(
        "\n".join("file '" + p.name + "'" for p in segments) + "\n", encoding="utf-8"
    )
    picture = output_dir / "picture.mp4"
    run_ffmpeg(["-f", "concat", "-safe", "1", "-i", listing, "-c", "copy", picture])
    output = output_dir / "result.mp4"
    total = sum(e["frames"] for e in edits) / fps
    if music:
        fade = min(0.8, total / 3)
        run_ffmpeg(
            [
                "-i",
                picture,
                "-stream_loop",
                "-1",
                "-i",
                music,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-af",
                f"loudnorm=I=-20:TP=-2:LRA=7,afade=t=in:st=0:d=0.2,afade=t=out:st={total - fade}:d={fade}",
                "-t",
                total,
                "-movflags",
                "+faststart",
                output,
            ]
        )
    else:
        run_ffmpeg(["-i", picture, "-c", "copy", "-an", "-movflags", "+faststart", output])
    progress(80, "Checking the new video timing")
    actual, _, frame_count = metadata(output)
    expected_frames = sum(e["frames"] for e in edits)
    if frame_count != expected_frames:
        raise RuntimeError(
            f"Export frame count differs from the template: {frame_count} vs {expected_frames}."
        )
    boundary_score = None
    try:
        from beauty_video_templates.parsing.backends import shots

        detected = shots(output, backend="adaptive")
        expected = [round(s["start"] * fps) for s in template["slots"][1:]]
        boundary_score = event_f1([s["start_frame"] for s in detected["items"][1:]], expected, 2)
    except Exception:
        pass
    comparison = None
    if reference:
        progress(90, "Creating the side-by-side presentation video")
        comparison = output_dir / "comparison.mp4"
        label = Image.new("RGBA", (720, 50), (14, 18, 29, 230))
        draw = ImageDraw.Draw(label)
        font = ImageFont.truetype(str(find_font()), 19)
        draw.text((18, 12), "REFERENCE", font=font, fill="white")
        draw.text((378, 12), "NEW PRODUCT VIDEO", font=font, fill="white")
        banner = output_dir / "comparison_label.png"
        label.save(banner)
        vf = f"[0:v]scale=360:640:force_original_aspect_ratio=decrease,pad=360:640:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},setpts=PTS-STARTPTS[a];[1:v]scale=360:640,setsar=1,fps={fps},setpts=PTS-STARTPTS[b];[a][b]hstack=shortest=1[joined];[joined][2:v]overlay=0:0:shortest=1,format=yuv420p[out]"
        run_ffmpeg(
            [
                "-i",
                reference,
                "-i",
                output,
                "-loop",
                "1",
                "-i",
                banner,
                "-filter_complex",
                vf,
                "-map",
                "[out]",
                "-map",
                "1:a?",
                "-frames:v",
                expected_frames,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "21",
                "-threads",
                "2",
                "-c:a",
                "aac",
                "-t",
                total,
                "-movflags",
                "+faststart",
                comparison,
            ]
        )
    report = {
        "duration_seconds": actual["duration_seconds"],
        "expected_duration_seconds": total,
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "timing_error_seconds": abs(actual["duration_seconds"] - total),
        "unique_materials": len({x["asset_id"] for x in edits}),
        "boundary_agreement": boundary_score,
        "audio": "user-supplied music" if music else "silent",
        "reference_audio_reused": False,
        "transition_policy": "hard cuts",
        "semantic_status": "editable local CLIP estimates / user tags; not validated VLM semantics",
        "edits": edits,
        "warnings": [e["warning"] for e in edits if e["warning"]],
    }
    (output_dir / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
