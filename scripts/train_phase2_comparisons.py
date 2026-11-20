"""Train small reference-conditioned comparisons with video-disjoint splits."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from beauty_video_templates.parsing.media import frame, metadata, prepare_indexed_frames


class TransitionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(24, 2),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--max-per-video", type=int, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    refs = list(
        csv.DictReader(
            next(a.root.rglob("200条视频镜头与转场合并表.csv")).open(encoding="utf-8-sig")
        )
    )
    videos = {}
    for path in sorted(a.root.rglob("*.mp4")):
        videos.setdefault(path.stem.split("_")[0], path)
    ids = sorted(videos)[:30]
    split = {
        key: "train" if i < 21 else "validation" if i < 24 else "test" for i, key in enumerate(ids)
    }
    labels = {
        "极近景": 0,
        "近景": 1,
        "中近景": 2,
        "中景": 2,
        "全景": 3,
        "局部特写": 4,
        "产品/手部特写": 4,
    }
    cached = a.output / "training_features_source_frames.npz"
    if not cached.exists():
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", use_safetensors=True
        ).eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        pairs = []
        targets = []
        groups = []
        features = []
        sizes = []
        size_groups = []
        rules = []
        for key in ids:
            path = videos[key]
            rows = [r for r in refs if r["样本编号"] == key]
            candidates = [r for r in rows if r["入镜转场序号"]]
            # Evenly spaced, fixed selection; no label-based cherry-picking.
            chosen = (
                np.linspace(
                    0, len(candidates) - 1, min(a.max_per_video, len(candidates)), dtype=int
                )
                if candidates
                else []
            )
            frame_count = metadata(path)[2]
            requested = []
            for index in chosen:
                boundary = int(candidates[index]["起始帧"])
                requested.extend(
                    min(max(0, boundary + offset), frame_count - 1) for offset in [-3, -1, 0, 2]
                )
            requested.extend(
                (int(r["起始帧"]) + int(r["结束帧"])) // 2 for r in rows if r["景别"] in labels
            )
            pts = prepare_indexed_frames(path, requested)
            for index in chosen:
                r = candidates[index]
                boundary = int(r["起始帧"])
                before = cv2.resize(frame(path, pts[max(0, boundary - 1)]), (32, 32))
                after = cv2.resize(frame(path, pts[boundary]), (32, 32))
                pairs.append(np.concatenate([before, after], axis=2).transpose(2, 0, 1) / 255)
                targets.append(0 if r["入镜转场类型"] in {"硬切", "跳切/动作切"} else 1)
                groups.append(key)
                images = [
                    frame(path, pts[min(max(0, boundary + offset), frame_count - 1)], 160)
                    for offset in [-3, -1, 0, 2]
                ]
                x = [
                    float(np.mean(np.abs(a.astype(float) - b.astype(float))) / 255)
                    for a, b in zip(images, images[1:])
                ]
                rules.append(0 if x[1] > max(x[0], x[2]) * 1.5 and x[1] > 0.08 else 1)
            eligible = [r for r in rows if r["景别"] in labels]
            chosen = (
                np.linspace(0, len(eligible) - 1, min(4, len(eligible)), dtype=int)
                if eligible
                else []
            )
            for index in chosen:
                r = eligible[index]
                t = pts[(int(r["起始帧"]) + int(r["结束帧"])) // 2]
                inputs = processor(
                    images=Image.fromarray(frame(path, t)[:, :, ::-1]), return_tensors="pt"
                )
                with torch.inference_mode():
                    x = model.get_image_features(**inputs).numpy()[0]
                features.append(x / np.linalg.norm(x))
                sizes.append(labels[r["景别"]])
                size_groups.append(key)
            print("training features", key, flush=True)
        np.savez_compressed(
            cached,
            pairs=np.asarray(pairs, dtype=np.float32),
            targets=targets,
            groups=groups,
            features=features,
            sizes=sizes,
            size_groups=size_groups,
            rules=rules,
        )
    data = np.load(cached)

    def mask(groups, part):
        return np.array([split[str(key)] == part for key in groups])

    train = mask(data["groups"], "train")
    val = mask(data["groups"], "validation")
    test = mask(data["groups"], "test")
    # Feature extraction constructs CLIP and consumes RNG; reseed at training start.
    torch.manual_seed(42)
    net = TransitionCNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    x = torch.from_numpy(data["pairs"])
    y = torch.tensor(data["targets"], dtype=torch.long)
    best = -1
    state = None
    for epoch in range(15):
        net.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(net(x[train]), y[train])
        loss.backward()
        optimizer.step()
        net.eval()
        with torch.inference_mode():
            pred = net(x[val]).argmax(1).numpy()
        score = f1_score(y[val], pred, average="macro", zero_division=0)
        if score > best:
            best = score
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    torch.save(state, a.output / "transition_cnn.pt")
    with torch.inference_mode():
        prediction = net(x[test]).argmax(1).numpy()
    summary = {
        "seed": 42,
        "split": split,
        "sampling": "up to 8 evenly spaced reference transitions and 4 reference shots per video",
        "scope": "reference-conditioned pilot, not independent end-to-end performance",
        "transition_labels": ["hard_cut", "non_hard_cut"],
        "cnn_epochs": 15,
        "validation_selection": "best macro F1, test used only for final reporting",
    }

    def metrics(truth, pred):
        return {
            "n": len(truth),
            "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(truth, pred).tolist(),
            "confusion_matrix_label_order": sorted(
                {int(x) for x in truth} | {int(x) for x in pred}
            ),
        }

    summary["transition_cnn"] = metrics(data["targets"][test], prediction)
    summary["transition_test_class_counts"] = {
        str(label): int((data["targets"][test] == label).sum()) for label in [0, 1]
    }
    summary["transition_rules"] = metrics(data["targets"][test], data["rules"][test])
    majority = int(np.bincount(data["targets"][train]).argmax())
    summary["transition_majority_baseline"] = metrics(
        data["targets"][test], np.full(test.sum(), majority)
    )
    st = mask(data["size_groups"], "train")
    sv = mask(data["size_groups"], "validation")
    se = mask(data["size_groups"], "test")
    best = -1
    probe = None
    for c in [0.01, 0.1, 1, 10]:
        candidate = LogisticRegression(C=c, max_iter=1000, random_state=42).fit(
            data["features"][st], data["sizes"][st]
        )
        score = accuracy_score(data["sizes"][sv], candidate.predict(data["features"][sv]))
        if score > best:
            best = score
            probe = candidate
            summary["linear_probe_C"] = c
    summary["shot_size_linear_probe"] = metrics(
        data["sizes"][se], probe.predict(data["features"][se])
    )
    summary["shot_size_labels"] = [
        "extreme_close_up",
        "close_up",
        "medium_shot",
        "wide_shot",
        "detail_shot",
    ]
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True
    ).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text = clip_processor(
        text=[
            "an extreme close-up of a face, only eyes or lips visible",
            "a close-up portrait showing a person's face and shoulders",
            "a medium shot showing a person from the waist up",
            "a wide shot showing a full person and surroundings",
            "a detail shot of a beauty product, hand or small object",
        ],
        return_tensors="pt",
        padding=True,
    )
    with torch.inference_mode():
        embeddings = clip_model.get_text_features(**text).numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    zero_shot = (data["features"][se] @ embeddings.T).argmax(1)
    summary["shot_size_zero_shot_same_test"] = metrics(data["sizes"][se], zero_shot)
    # Store numeric coefficients; avoid pickle deserialization for downstream inference.
    np.savez(
        a.output / "shot_size_probe.npz",
        coef=probe.coef_,
        intercept=probe.intercept_,
        classes=probe.classes_,
    )
    (a.output / "learned_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
