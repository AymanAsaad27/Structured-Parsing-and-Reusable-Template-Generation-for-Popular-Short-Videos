"""Freeze a new video-disjoint protocol and extract resumable full-dataset features."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from beauty_video_templates.parsing.media import frame, prepare_indexed_frames

SIZE = {
    "极近景": 0,
    "近景": 1,
    "中近景": 2,
    "中景": 2,
    "全景": 3,
    "局部特写": 4,
    "产品/手部特写": 4,
}
TRANSITION = {"硬切": 0, "跳切/动作切": 1, "叠化/渐变转场": 2, "闪黑/黑场转场": 3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    a = p.parse_args()
    if a.shards < 1 or not 0 <= a.shard < a.shards:
        p.error("Require shards >= 1 and 0 <= shard < shards")
    a.output.mkdir(parents=True, exist_ok=True)
    source = next(a.root.rglob("200条视频镜头与转场合并表.csv"))
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig")))
    videos = {}
    for path in sorted(a.root.rglob("*.mp4")):
        videos.setdefault(path.stem.split("_")[0], path)
    ids = sorted({r["样本编号"] for r in rows} & set(videos))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol_file = a.output / "protocol.json"
    if not protocol_file.exists():
        unseen = [v for v in ids if int(v[1:]) > 30]
        rng = np.random.default_rng(42)
        rng.shuffle(unseen)
        # Cover the rare class in validation/test before any performance inspection.
        rare = sorted({r["样本编号"] for r in rows if r["入镜转场类型"] == "闪黑/黑场转场"})
        test = [rare[-1]] + [v for v in unseen if v not in rare][:39]
        remaining = [v for v in unseen if v not in test]
        val = [rare[-2]] + [v for v in remaining if v != rare[-2]][:19]
        split = {v: "test" if v in test else "validation" if v in val else "train" for v in ids}
        protocol = {
            "seed": 42,
            "annotation_sha256": digest,
            "split": split,
            "prior_pilot_ids": "V001-V030 restricted to training; the pilot was already inspected",
            "split_policy": "video-disjoint 140/20/40; rare-class coverage allocated before model performance evaluation",
            "creator_campaign_grouping": "not established; residual leakage risk must be audited",
            "reference_review_status": "user reports reviewed annotations; exact review provenance still requested",
            "transition_labels": ["hard_cut", "jump_cut", "dissolve", "black_transition"],
            "size_labels": [
                "extreme_close_up",
                "close_up",
                "medium_shot",
                "wide_shot",
                "detail_shot",
            ],
        }
        protocol_file.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    protocol = json.loads(protocol_file.read_text())
    if protocol["annotation_sha256"] != digest:
        raise ValueError("Annotation version changed; use a new experiment folder")
    torch.set_num_threads(2)
    local = Path("models/weights/clip-b32")
    model = CLIPModel.from_pretrained(local, use_safetensors=True, local_files_only=True).eval()
    processor = CLIPProcessor.from_pretrained(local, local_files_only=True)
    for position, key in enumerate(ids):
        if position % a.shards != a.shard:
            continue
        target = a.output / (key + ".npz")
        if target.exists():
            continue
        selected = [r for r in rows if r["样本编号"] == key]
        transitions = [r for r in selected if r["入镜转场类型"] in TRANSITION]
        sizes = [r for r in selected if r["景别"] in SIZE]
        indices = []
        for r in transitions:
            center = int(r["起始帧"])
            indices.extend(max(0, center + i) for i in [-4, -1, 0, 2])
        indices.extend((int(r["起始帧"]) + int(r["结束帧"])) // 2 for r in sizes)
        pts = prepare_indexed_frames(videos[key], indices)
        pairs = []
        features = []
        labels = []
        for r in transitions:
            center = int(r["起始帧"])
            images = [
                cv2.resize(frame(videos[key], pts[min(max(0, center + i), len(pts) - 1)]), (32, 32))
                for i in [-4, -1, 0, 2]
            ]
            pairs.append(np.concatenate(images[1:3], axis=2).transpose(2, 0, 1) / 255)
            features.append(
                [
                    float(np.mean(np.abs(x.astype(float) - y.astype(float))) / 255)
                    for x, y in zip(images, images[1:])
                ]
            )
            labels.append(TRANSITION[r["入镜转场类型"]])
        embeddings = []
        for start in range(0, len(sizes), 8):
            images = [
                Image.fromarray(
                    frame(videos[key], pts[(int(r["起始帧"]) + int(r["结束帧"])) // 2])[:, :, ::-1]
                )
                for r in sizes[start : start + 8]
            ]
            inputs = processor(images=images, return_tensors="pt")
            with torch.inference_mode():
                x = model.get_image_features(**inputs).numpy()
            embeddings.extend(x / np.linalg.norm(x, axis=1, keepdims=True))
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            pairs=np.asarray(pairs, dtype=np.float32).reshape(-1, 6, 32, 32),
            transition_features=np.asarray(features, dtype=np.float32).reshape(-1, 3),
            transition_labels=np.asarray(labels, dtype=np.int64),
            size_features=np.asarray(embeddings, dtype=np.float32).reshape(-1, 512),
            size_labels=np.asarray([SIZE[r["景别"]] for r in sizes], dtype=np.int64),
        )
        temporary.replace(target)
        print(key, "transitions", len(labels), "size frames", len(sizes), flush=True)


if __name__ == "__main__":
    main()
