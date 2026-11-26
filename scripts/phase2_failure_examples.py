"""Render source-frame pairs for a reproducible boundary failure illustration."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from beauty_video_templates.parsing.media import frame, prepare_indexed_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("experiment", type=Path)
    p.add_argument("--video-id", default="V056")
    p.add_argument("--font", type=Path)
    a = p.parse_args()
    path = next(a.root.rglob(a.video_id + "_*.mp4"))
    rows = list(
        csv.DictReader(
            next(a.root.rglob("200条视频镜头与转场合并表.csv")).open(encoding="utf-8-sig")
        )
    )
    pred = json.loads((a.experiment / "boundaries" / (a.video_id + "_adaptive.json")).read_text())
    points = [x["start_frame"] for x in pred["items"][1:]]
    missed = [
        r
        for r in rows
        if r["样本编号"] == a.video_id
        and int(r["起始帧"]) > 2
        and not any(abs(int(r["起始帧"]) - t) <= 2 for t in points)
    ][:4]
    if len(missed) < 4 or not points:
        raise ValueError(
            "This illustration expects at least four missed boundaries and one prediction"
        )
    chosen = [missed[0], missed[3]]
    indices = [int(r["起始帧"]) + offset for r in chosen for offset in [-2, 2]]
    pts = prepare_indexed_frames(path, indices)
    image = Image.new("RGB", (900, 980), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(a.font), 19) if a.font else ImageFont.load_default(size=19)
    for row, reference in enumerate(chosen):
        center = int(reference["起始帧"])
        nearest = min(points, key=lambda t: abs(t - center))
        draw.text(
            (16, row * 490 + 5),
            f"{a.video_id}: reference boundary {center}; nearest adaptive cut {nearest}",
            fill="black",
            font=font,
        )
        for col, offset in enumerate([-2, 2]):
            source = Image.fromarray(frame(path, pts[center + offset])[:, :, ::-1])
            source.thumbnail((430, 425))
            image.paste(source, (col * 450 + (450 - source.width) // 2, row * 490 + 32))
            draw.text(
                (col * 450 + 16, row * 490 + 459),
                f"Source frame {center + offset}",
                fill="black",
                font=font,
            )
    image.save(a.experiment / "failure_examples.png")


if __name__ == "__main__":
    main()
