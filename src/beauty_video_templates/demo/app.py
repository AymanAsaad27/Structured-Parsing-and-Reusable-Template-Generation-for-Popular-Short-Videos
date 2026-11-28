"""Loopback-only FastAPI app for the research presentation studio."""

import argparse
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import AUDIO, IMAGE, VIDEO, Studio


class AnalysisRequest(BaseModel):
    read_text: bool = True
    detect_beats: bool = True


class TemplateRequest(BaseModel):
    template: dict


class RenderRequest(TemplateRequest):
    product_name: str = Field(default="Your product", max_length=100)
    selling_points: list[str] = Field(default_factory=list, max_length=6)
    captions: bool = True


class AssetRequest(BaseModel):
    role: str


def create_app(root=None, repo=None, preset=None):
    repo = Path(repo or os.environ.get("BVT_REPO_ROOT", Path.cwd())).resolve()
    studio = Studio(root or repo / "artifacts/demo/sessions", repo, preset)
    app = FastAPI(title="Beauty Template Studio", version="0.1.0")
    app.state.studio = studio

    @app.middleware("http")
    async def local_client(request: Request, call_next):
        if (
            request.method in {"POST", "PUT", "PATCH", "DELETE"}
            and request.headers.get("x-demo-client") != "1"
        ):
            return JSONResponse(
                {"detail": "Use the local studio interface for this action."}, status_code=403
            )
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_request(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.get("/api/health")
    def health():
        return {
            "status": "ready",
            "sample_available": studio.sample_available(),
            "visual_model_available": (studio.model_dir / "model.safetensors").is_file(),
            "roles": [
                "hook",
                "product_introduction",
                "texture_demonstration",
                "application_demonstration",
                "result_comparison",
                "presenter_explanation",
                "call_to_action",
                "detail_cutaway",
            ],
        }

    @app.post("/api/sessions")
    def new_session():
        return studio.public(studio.create())

    @app.post("/api/sample")
    def sample():
        identifier = studio.create_sample()
        return studio.public(identifier)

    @app.get("/api/sessions/{identifier}")
    def session(identifier: str):
        return studio.public(identifier)

    @app.post("/api/sessions/{identifier}/upload/{kind}")
    async def upload(identifier: str, kind: str, files: list[UploadFile]):
        studio.idle(identifier)
        if kind not in {"reference", "assets", "music"}:
            raise HTTPException(404, "Unknown upload category.")
        if kind != "assets" and len(files) != 1:
            raise ValueError("Choose one reference video or one music file at a time.")
        if not files or len(files) > 20:
            raise ValueError("Choose between 1 and 20 files.")
        allowed = VIDEO if kind == "reference" else AUDIO if kind == "music" else VIDEO | IMAGE
        limit = 150 * 1024 * 1024 if kind == "reference" else 60 * 1024 * 1024
        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in allowed:
                raise ValueError("Unsupported file type for this upload.")
            target = studio.directory(identifier) / "uploads" / (uuid.uuid4().hex + suffix)
            try:
                size = 0
                with target.open("wb") as stream:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(
                                f"File is too large. Limit: {limit // 1024 // 1024} MB."
                            )
                        stream.write(chunk)
                studio.add_file(
                    identifier,
                    target,
                    file.filename or target.name,
                    "asset" if kind == "assets" else kind,
                )
            except Exception as error:
                target.unlink(missing_ok=True)
                if isinstance(error, ValueError):
                    raise
                raise ValueError(
                    "This file could not be decoded. Choose a supported image, video or audio file."
                ) from error
            finally:
                await file.close()
        return studio.public(identifier)

    @app.patch("/api/sessions/{identifier}/assets/{asset_id}")
    def tag_asset(identifier: str, asset_id: str, body: AssetRequest):
        studio.set_asset_role(identifier, asset_id, body.role)
        return studio.public(identifier)

    @app.delete("/api/sessions/{identifier}/assets/{asset_id}")
    def remove_asset(identifier: str, asset_id: str):
        studio.remove_asset(identifier, asset_id)
        return studio.public(identifier)

    @app.put("/api/sessions/{identifier}/template")
    def put_template(identifier: str, body: TemplateRequest):
        studio.set_template(identifier, body.template)
        return studio.public(identifier)

    @app.get("/api/sessions/{identifier}/template")
    def get_template(identifier: str):
        data = studio.load(identifier)
        if not data["template"]:
            raise ValueError("Analyse a reference or import a template first.")
        return JSONResponse(
            data["template"],
            headers={"Content-Disposition": 'attachment; filename="beauty-template.json"'},
        )

    @app.post("/api/sessions/{identifier}/analyse")
    def analyse(identifier: str, body: AnalysisRequest):
        return {"job_id": studio.start_analysis(identifier, body.read_text, body.detect_beats)}

    @app.post("/api/sessions/{identifier}/render")
    def render(identifier: str, body: RenderRequest):
        return {
            "job_id": studio.start_render(
                identifier, body.template, body.product_name, body.selling_points, body.captions
            )
        }

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        with studio.lock:
            if job_id not in studio.jobs:
                raise HTTPException(404, "Job not found. Refresh the studio session.")
            return dict(studio.jobs[job_id])

    @app.get("/api/sessions/{identifier}/media/{name:path}")
    def media(identifier: str, name: str):
        return FileResponse(studio.owned_media(identifier, name))

    static = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static, html=True), name="studio")
    return app


def main():
    parser = argparse.ArgumentParser(description="Run the local Beauty Template Studio")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    import uvicorn

    print(f"Open http://127.0.0.1:{args.port} to use Beauty Template Studio.", flush=True)
    uvicorn.run(create_app(repo=args.repo), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
