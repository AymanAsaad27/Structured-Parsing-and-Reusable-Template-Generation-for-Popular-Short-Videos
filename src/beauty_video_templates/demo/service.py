"""Persistent local sessions and serial background work for the video studio."""

import copy
import hashlib
import json
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from beauty_video_templates.semantics.clip_roles import ROLES
from beauty_video_templates.templates.demo_template import validate_template

VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Studio:
    def __init__(self, root, repo, preset=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo = Path(repo).resolve()
        self.model_dir = self.repo / "models/weights/clip-b32"
        self.preset = Path(preset) if preset else self.repo / "artifacts/demo/preset.json"
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-studio")
        self.jobs = {}
        self.busy = set()

    def directory(self, session_id):
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise ValueError("Unknown studio session.")
        path = self.root / session_id
        if not (path / "session.json").is_file():
            raise ValueError("This studio session is no longer available.")
        return path

    def create(self):
        identifier = uuid.uuid4().hex
        directory = self.root / identifier
        directory.mkdir()
        (directory / "uploads").mkdir()
        self.save(
            {
                "id": identifier,
                "reference": None,
                "assets": [],
                "music": None,
                "template": None,
                "warnings": [],
                "last_render": None,
                "product_name": "Medicube Collagen Night Wrapping Mask",
                "selling_points": [],
            }
        )
        return identifier

    def save(self, data):
        with self.lock:
            path = self.root / data["id"] / "session.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)

    def load(self, identifier):
        with self.lock:
            return json.loads(
                (self.directory(identifier) / "session.json").read_text(encoding="utf-8")
            )

    def idle(self, identifier):
        with self.lock:
            if identifier in self.busy:
                raise ValueError("Please wait for the current analysis or render to finish.")

    def public(self, identifier):
        data = self.load(identifier)
        for category in ["reference", "music"]:
            if data[category]:
                data[category] = {
                    k: v for k, v in data[category].items() if k not in {"path", "sha256"}
                }
        data["assets"] = [
            {
                k: v
                for k, v in asset.items()
                if k not in {"path", "sha256", "role_scores", "size_scores"}
            }
            for asset in data["assets"]
        ]
        return data

    def url(self, identifier, path):
        relative = Path(path).resolve().relative_to(self.directory(identifier))
        return f"/api/sessions/{identifier}/media/" + relative.as_posix()

    def owned_media(self, identifier, name):
        root = self.directory(identifier)
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.suffix.lower() not in VIDEO | IMAGE | AUDIO | {".json"}
        ):
            raise ValueError("Media file is unavailable.")
        # Never serve session internals or upload metadata as arbitrary JSON.
        if path.suffix == ".json" and path.name not in {"template.json", "evidence.json"}:
            raise ValueError("This file is not a downloadable export.")
        return path

    def add_file(self, identifier, path, name, kind, role="auto"):
        self.idle(identifier)
        data = self.load(identifier)
        path = Path(path)
        identifier_asset = uuid.uuid4().hex[:12]
        item = {
            "id": identifier_asset,
            "name": Path(name).name,
            "path": str(path.resolve()),
            "url": self.url(identifier, path),
            "sha256": file_hash(path),
        }
        if kind == "reference":
            from beauty_video_templates.parsing.media import metadata

            info, _, _ = metadata(path)
            if not 1 <= info["duration_seconds"] <= 90:
                raise ValueError("Choose a reference between 1 and 90 seconds long.")
            item["duration_seconds"] = info["duration_seconds"]
            data["reference"] = item
            data["template"] = None
            data["last_render"] = None
        elif kind == "music":
            # FFmpeg will decode the file for rendering; preflight catches invalid audio.
            import av

            with av.open(str(path)) as opened:
                if not opened.streams.audio:
                    raise ValueError("This file has no audio stream.")
            data["music"] = item
        else:
            if len(data["assets"]) >= 20:
                raise ValueError("This demo supports up to 20 product materials per session.")
            if role != "auto" and role not in ROLES:
                raise ValueError("Unknown material tag.")
            item.update({"role": role, "analysed": False})
            # Validate decoding now; visual matching runs in the background task.
            if path.suffix.lower() in IMAGE:
                from PIL import Image

                with Image.open(path) as image:
                    image.verify()
                item["kind"] = "image"
                item["duration_seconds"] = None
            else:
                from beauty_video_templates.parsing.media import metadata

                item["kind"] = "video"
                item["duration_seconds"] = metadata(path)[0]["duration_seconds"]
            data["assets"].append(item)
        self.save(data)
        return item

    def set_asset_role(self, identifier, asset_id, role):
        self.idle(identifier)
        if role != "auto" and role not in ROLES:
            raise ValueError("Unknown material tag.")
        data = self.load(identifier)
        found = next((x for x in data["assets"] if x["id"] == asset_id), None)
        if not found:
            raise ValueError("Material not found.")
        found["role"] = role
        self.save(data)

    def remove_asset(self, identifier, asset_id):
        self.idle(identifier)
        data = self.load(identifier)
        data["assets"] = [a for a in data["assets"] if a["id"] != asset_id]
        self.save(data)

    def set_template(self, identifier, template):
        self.idle(identifier)
        validate_template(template)
        data = self.load(identifier)
        data["template"] = copy.deepcopy(template)
        data["last_render"] = None
        self.save(data)

    def job(self, identifier, kind, work):
        with self.lock:
            self.idle(identifier)
            self.directory(identifier)
            self.busy.add(identifier)
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {
                "id": job_id,
                "session_id": identifier,
                "kind": kind,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "created_at": time.time(),
            }

        def progress(value, message):
            with self.lock:
                self.jobs[job_id].update(progress=value, message=message)

        def execute():
            try:
                with self.lock:
                    self.jobs[job_id]["status"] = "running"
                work(progress)
                with self.lock:
                    self.jobs[job_id].update(status="succeeded", progress=100, message="Ready")
            except Exception as error:
                (self.directory(identifier) / ("error_" + job_id + ".log")).write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                with self.lock:
                    self.jobs[job_id].update(
                        status="failed",
                        message=str(error)[:1200]
                        if isinstance(error, ValueError)
                        else "This step could not finish. Check the selected media and try again; diagnostic details were saved with the session.",
                    )
            finally:
                with self.lock:
                    self.busy.discard(identifier)

        self.executor.submit(execute)
        return job_id

    def start_analysis(self, identifier, read_text=True, detect_beats=True):
        data = self.load(identifier)
        if not data["reference"]:
            raise ValueError("Upload a reference video first.")

        def work(progress):
            from beauty_video_templates.pipelines.demo_analysis import analyse_reference

            template, warnings = analyse_reference(
                data["reference"]["path"],
                self.directory(identifier) / "analysis",
                self.model_dir,
                read_text=read_text,
                detect_beats=detect_beats,
                progress=progress,
            )
            fresh = self.load(identifier)
            template["provenance"]["reference_asset_id"] = data["reference"]["id"]
            fresh.update(template=template, warnings=warnings, last_render=None)
            self.save(fresh)

        return self.job(identifier, "analysis", work)

    def start_render(self, identifier, template, product_name, points, captions=True):
        validate_template(template)
        data = self.load(identifier)
        if not data["assets"]:
            raise ValueError("Add product materials before creating the video.")
        if len(product_name) > 100 or len(points) > 6 or any(len(p) > 180 for p in points):
            raise ValueError(
                "Keep the product name under 100 characters and add up to six short talking points."
            )
        template = copy.deepcopy(template)

        def work(progress):
            from beauty_video_templates.matching.demo_matcher import assign_materials
            from beauty_video_templates.pipelines.demo_analysis import (
                analyse_material,
                fill_captions,
            )
            from beauty_video_templates.rendering.demo_renderer import render_demo

            for i, asset in enumerate(data["assets"]):
                if not asset.get("analysed"):
                    thumb = self.directory(identifier) / "uploads" / (asset["id"] + "_thumb.jpg")
                    estimate = analyse_material(asset["path"], self.model_dir, thumb)
                    asset.update(estimate, analysed=True, thumbnail_url=self.url(identifier, thumb))
                progress(
                    int(8 * (i + 1) / len(data["assets"])), f"Matching product material {i + 1}"
                )
            fill_captions(template, product_name, points)
            # Preserve material estimates if a later FFmpeg step fails or is interrupted.
            saved = self.load(identifier)
            saved["assets"] = data["assets"]
            self.save(saved)
            assignments = assign_materials(template, data["assets"])
            render_id = uuid.uuid4().hex[:12]
            output = self.directory(identifier) / "renders" / render_id
            evidence = render_demo(
                template,
                assignments,
                data["assets"],
                output,
                music=data["music"]["path"] if data["music"] else None,
                captions=captions,
                reference=data["reference"]["path"] if data["reference"] else None,
                progress=progress,
            )
            hashes = {
                x["sha256"]
                for x in data["assets"]
                if x["id"] in {a["asset_id"] for a in assignments}
            }
            evidence["identical_reference_asset_selected"] = bool(
                data["reference"] and data["reference"]["sha256"] in hashes
            )
            evidence["content_check_note"] = (
                "File identity check only; not a visual originality assessment."
            )
            (output / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            data.update(
                template=template,
                product_name=product_name,
                selling_points=points,
                last_render={
                    "id": render_id,
                    "video_url": self.url(identifier, output / "result.mp4"),
                    "comparison_url": self.url(identifier, output / "comparison.mp4")
                    if (output / "comparison.mp4").is_file()
                    else None,
                    "template_url": self.url(identifier, output / "template.json"),
                    "evidence_url": self.url(identifier, output / "evidence.json"),
                    "evidence": evidence,
                },
            )
            self.save(data)

        return self.job(identifier, "render", work)

    def sample_available(self):
        if not self.preset.is_file():
            return False
        try:
            config = json.loads(self.preset.read_text(encoding="utf-8"))
            return Path(config["reference"]).is_file() and all(
                Path(a["path"]).is_file() for a in config["assets"]
            )
        except (ValueError, KeyError, TypeError):
            return False

    def create_sample(self):
        if not self.sample_available():
            raise ValueError(
                "Sample media is not configured. Upload your own files, or run scripts/prepare_studio_demo.py."
            )
        config = json.loads(self.preset.read_text(encoding="utf-8"))
        identifier = self.create()

        def copy_file(source):
            source = Path(source)
            target = (
                self.directory(identifier) / "uploads" / (uuid.uuid4().hex + source.suffix.lower())
            )
            shutil.copyfile(source, target)
            return target

        self.add_file(
            identifier, copy_file(config["reference"]), Path(config["reference"]).name, "reference"
        )
        for asset in config["assets"]:
            self.add_file(
                identifier,
                copy_file(asset["path"]),
                asset.get("name", Path(asset["path"]).name),
                "asset",
                asset.get("role", "auto"),
            )
        if config.get("music") and Path(config["music"]).is_file():
            self.add_file(
                identifier, copy_file(config["music"]), Path(config["music"]).name, "music"
            )
        data = self.load(identifier)
        data["product_name"] = config.get("product_name", "Medicube Collagen Night Wrapping Mask")
        data["selling_points"] = config.get("selling_points", [])
        self.save(data)
        return identifier
