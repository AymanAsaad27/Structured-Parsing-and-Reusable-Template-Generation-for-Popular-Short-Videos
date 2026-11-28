"""Configure local sample media; never submits a new Seedance task."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--sample-id", default="V005")
    p.add_argument(
        "--clips",
        type=Path,
        required=True,
        help="Folder containing the four existing Medicube clips",
    )
    p.add_argument("--music", type=Path)
    p.add_argument("--output", type=Path, default=Path("artifacts/demo/preset.json"))
    a = p.parse_args()
    candidates = sorted(a.resources.rglob(a.sample_id + "_*.mp4"))
    if not candidates:
        raise FileNotFoundError("Reference sample ID not found.")
    definitions = [
        ("medicube-hero.mp4", "Medicube - product reveal", "product_introduction"),
        ("medicube-texture.mp4", "Medicube - texture detail", "texture_demonstration"),
        (
            "medicube-application_open.mp4",
            "Medicube - application opening",
            "application_demonstration",
        ),
        (
            "medicube-application_detail.mp4",
            "Medicube - application detail",
            "application_demonstration",
        ),
    ]
    assets = []
    for name, label, role in definitions:
        path = (a.clips / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        assets.append(
            {
                "path": str(path),
                "name": label,
                "role": role,
                "origin": "previously generated Seedance footage",
            }
        )
    config = {
        "reference": str(candidates[0].resolve()),
        "assets": assets,
        "music": str(a.music.resolve()) if a.music else None,
        "product_name": "Medicube Collagen Night Wrapping Mask",
        "selling_points": [],
        "note": "Existing generated material reused for a new automatically extracted reference template; no new paid generation.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Prepared local Medicube sample:", a.output)


if __name__ == "__main__":
    main()
