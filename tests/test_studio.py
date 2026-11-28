"""Meaningful prototype checks: isolation, matching, timing, and real media export."""

# ruff: noqa: E402

import copy
import io
import threading
import time
import wave

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("av")
from fastapi.testclient import TestClient
from PIL import Image

from beauty_video_templates.demo.app import create_app
from beauty_video_templates.matching.demo_matcher import assign_materials
from beauty_video_templates.pipelines.demo_analysis import analyse_reference
from beauty_video_templates.rendering.demo_renderer import render_demo
from beauty_video_templates.templates.demo_template import validate_template


def example_template():
    return {
        "schema_version": "1.0-demo",
        "template_id": "example",
        "width": 360,
        "height": 640,
        "fps": 24,
        "duration_seconds": 2,
        "style": {"palette": ["#eee8dd"]},
        "slots": [
            {
                "id": "a",
                "start": 0,
                "end": 1,
                "role": "product_introduction",
                "caption": "A product, not copied text",
                "caption_box": [0.1, 0.7, 0.9, 0.85],
            },
            {
                "id": "b",
                "start": 1,
                "end": 2,
                "role": "texture_demonstration",
                "caption": "Texture illustration",
                "caption_box": [0.1, 0.7, 0.9, 0.85],
            },
        ],
    }


def test_template_rejects_gap_nan_and_invalid_box():
    t = example_template()
    validate_template(t)
    for key, value in [
        ("start", 0.2),
        ("end", float("nan")),
        ("caption_box", [0.9, 0.7, 0.1, 0.8]),
    ]:
        altered = copy.deepcopy(t)
        altered["slots"][0][key] = value
        with pytest.raises(ValueError):
            validate_template(altered)


def test_matching_follows_material_tags_and_manual_override():
    assets = [
        {"id": "hero", "name": "Product", "role": "product_introduction"},
        {"id": "texture", "name": "Texture", "role": "texture_demonstration"},
    ]
    template = example_template()
    assert [x["asset_id"] for x in assign_materials(template, assets)] == ["hero", "texture"]
    template["slots"][0]["asset_id"] = "texture"
    assert assign_materials(template, assets)[0]["selection"] == "manual"
    template["slots"][0]["asset_id"] = "removed"
    with pytest.raises(ValueError):
        assign_materials(template, assets)


def test_api_upload_import_and_media_isolation(tmp_path):
    app = create_app(tmp_path / "sessions", tmp_path)
    with TestClient(app) as client:
        assert client.post("/api/sessions").status_code == 403
        headers = {"X-Demo-Client": "1"}
        data = client.post("/api/sessions", headers=headers).json()
        sid = data["id"]
        payload = io.BytesIO()
        Image.new("RGB", (80, 120), "yellow").save(payload, format="PNG")
        response = client.post(
            f"/api/sessions/{sid}/upload/assets",
            headers=headers,
            files=[("files", ("product.png", payload.getvalue(), "image/png"))],
        )
        assert response.status_code == 200
        asset = response.json()["assets"][0]
        assert "path" not in asset and "sha256" not in asset
        assert client.get(asset["url"]).status_code == 200
        assert client.get(f"/api/sessions/{sid}/media/session.json").status_code == 400
        assert client.get(f"/api/sessions/{sid}/media/%2E%2E/%2E%2E/secret.mp4").status_code == 400
        bad = client.post(
            f"/api/sessions/{sid}/upload/assets",
            headers=headers,
            files=[("files", ("corrupt.png", b"not an image", "image/png"))],
        )
        assert bad.status_code == 400
        imported = client.put(
            f"/api/sessions/{sid}/template", headers=headers, json={"template": example_template()}
        )
        assert imported.status_code == 200
        assert client.get(f"/api/sessions/{sid}/template").json()["duration_seconds"] == 2
        assert (
            client.post(f"/api/sessions/{sid}/analyse", headers=headers, json={}).status_code == 400
        )
    app.state.studio.executor.shutdown(wait=True)


def test_session_blocks_mutation_during_background_work(tmp_path):
    app = create_app(tmp_path / "sessions", tmp_path)
    studio = app.state.studio
    sid = studio.create()
    release = threading.Event()
    job = studio.job(sid, "test", lambda progress: release.wait(5))
    try:
        with pytest.raises(ValueError):
            studio.set_template(sid, example_template())
        assert studio.jobs[job]["session_id"] == sid
    finally:
        release.set()
        studio.executor.shutdown(wait=True)
    assert studio.jobs[job]["status"] == "succeeded"


def test_imported_template_renders_new_uploads_without_reference(tmp_path):
    app = create_app(tmp_path / "sessions", tmp_path)
    headers = {"X-Demo-Client": "1"}
    with TestClient(app) as client:
        sid = client.post("/api/sessions", headers=headers).json()["id"]
        for colour, role in [
            ("yellow", "product_introduction"),
            ("green", "texture_demonstration"),
        ]:
            image = io.BytesIO()
            Image.new("RGB", (100, 180), colour).save(image, format="PNG")
            data = client.post(
                f"/api/sessions/{sid}/upload/assets",
                headers=headers,
                files=[("files", (colour + ".png", image.getvalue(), "image/png"))],
            ).json()
            asset_id = data["assets"][-1]["id"]
            assert (
                client.patch(
                    f"/api/sessions/{sid}/assets/{asset_id}", headers=headers, json={"role": role}
                ).status_code
                == 200
            )
        template = example_template()
        for slot in template["slots"]:
            slot["caption"] = ""
        assert (
            client.put(
                f"/api/sessions/{sid}/template", headers=headers, json={"template": template}
            ).status_code
            == 200
        )
        job = client.post(
            f"/api/sessions/{sid}/render",
            headers=headers,
            json={"template": template, "product_name": "A different product"},
        ).json()["job_id"]
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            status = client.get("/api/jobs/" + job).json()
            if status["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)
        assert status["status"] == "succeeded", status
        data = client.get(f"/api/sessions/{sid}").json()
        assert data["reference"] is None
        assert data["template"]["slots"][0]["caption"] == "A different product"
        assert data["last_render"]["comparison_url"] is None
        assert client.get(data["last_render"]["video_url"]).headers["content-type"] == "video/mp4"
    app.state.studio.executor.shutdown(wait=True)


def test_analysis_uses_detected_boundaries_without_annotation_lookup(tmp_path):
    pytest.importorskip("scenedetect")
    path = tmp_path / "reference.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24, (160, 240))
    if not writer.isOpened():
        pytest.skip("MJPG codec unavailable")
    for colour in [(0, 0, 255), (255, 0, 0)]:
        for _ in range(24):
            image = np.zeros((240, 160, 3), dtype=np.uint8)
            image[:] = colour
            writer.write(image)
    writer.release()
    template, warnings = analyse_reference(
        path, tmp_path / "analysis", tmp_path / "no-model", read_text=False, detect_beats=False
    )
    assert len(template["slots"]) == 2
    assert template["slots"][1]["start"] == pytest.approx(1)
    assert template["slots"][0]["role"] == "hook"
    assert warnings


@pytest.mark.parametrize("with_music", [False, True])
def test_real_renderer_preserves_frame_count_with_new_images(tmp_path, with_music):
    pytest.importorskip("imageio_ffmpeg")
    assets = []
    for key, colour, role in [
        ("hero", "gold", "product_introduction"),
        ("texture", "green", "texture_demonstration"),
    ]:
        path = tmp_path / (key + ".png")
        Image.new("RGB", (160, 240), colour).save(path)
        assets.append(
            {
                "id": key,
                "name": key,
                "path": str(path),
                "kind": "image",
                "duration_seconds": None,
                "role": role,
            }
        )
    template = example_template()
    assignments = assign_materials(template, assets)
    music = reference = None
    if with_music:
        music = tmp_path / "music.wav"
        samples = (np.sin(np.arange(3 * 22050) * 2 * np.pi * 330 / 22050) * 3000).astype("<i2")
        with wave.open(str(music), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(22050)
            audio.writeframes(samples.tobytes())
        reference = tmp_path / "reference.avi"
        writer = cv2.VideoWriter(str(reference), cv2.VideoWriter_fourcc(*"MJPG"), 24, (160, 240))
        for _ in range(48):
            writer.write(np.full((240, 160, 3), 120, dtype=np.uint8))
        writer.release()
    result = render_demo(
        template, assignments, assets, tmp_path / "export", music=music, reference=reference
    )
    assert result["frames"] == 48
    assert result["duration_seconds"] == pytest.approx(2, abs=1 / 24)
    assert result["reference_audio_reused"] is False
    assert result["unique_materials"] == 2
    assert (tmp_path / "export/result.mp4").stat().st_size > 1000
    if with_music:
        import av

        with av.open(str(tmp_path / "export/result.mp4")) as exported:
            assert len(exported.streams.audio) == 1
        with av.open(str(tmp_path / "export/comparison.mp4")) as compared:
            assert compared.streams.video[0].frames == 48
    with pytest.raises(ValueError):
        render_demo(template, assignments[:1], assets, tmp_path / "bad")


def test_short_video_extends_without_looping_back_inside_a_slot(tmp_path):
    source = tmp_path / "moving.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 24, (160, 240))
    if not writer.isOpened():
        pytest.skip("MJPG codec unavailable")
    for i in range(24):
        image = np.zeros((240, 160, 3), dtype=np.uint8)
        image[:, :, 2] = i * 10
        writer.write(image)
    writer.release()
    template = example_template()
    template["slots"] = template["slots"][:1]
    template["slots"][0]["end"] = 3
    template["duration_seconds"] = 3
    assets = [
        {
            "id": "clip",
            "name": "Moving product shot",
            "path": str(source),
            "kind": "video",
            "duration_seconds": 1,
            "role": "product_introduction",
        }
    ]
    result = render_demo(
        template, assign_materials(template, assets), assets, tmp_path / "extended", captions=False
    )
    assert result["frames"] == 72
    cap = cv2.VideoCapture(str(tmp_path / "extended/result.mp4"))
    values = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        values.append(int(image[image.shape[0] // 2, image.shape[1] // 2, 2]))
    cap.release()
    assert values[0] < 30 and values[-1] > 190
    assert min(np.diff(values)) > -30
    assert result["edits"][0]["looped"] is False
