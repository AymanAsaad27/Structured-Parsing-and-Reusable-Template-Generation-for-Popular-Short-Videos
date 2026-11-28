# Beauty Template Studio: presentation prototype

The local studio demonstrates **reference upload -> automatically extracted editing template -> new product materials -> rendered MP4**. It builds on the six Phase 2 adapters and the earlier Medicube clips; it does not use the supplied shot annotation table to manufacture an analysis result.

## Start

From the repository root:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,demo,phase2]"
.venv\Scripts\python -m beauty_video_templates.demo.app
```

Open `http://127.0.0.1:8765`. On Windows, double-click `Start_Studio.bat` to start the same server; `Start_Studio.ps1` is an alternative. `bvt-studio` is also available after editable installation. The server binds only to this computer. Sessions and uploads stay in ignored `artifacts/demo/sessions`; the studio does not read the Ark key or submit paid generation jobs.

The visual estimator uses the existing local CLIP ViT-B/32 model at `models/weights/clip-b32`. Without it or the vision dependencies, timing extraction still runs and the UI explains that the shot labels need manual review. OCR and beat estimation are optional; their dependencies come with the Phase 2 installation. No VLM API integration is claimed.

## Configure the Medicube sample

The demo package contains previously generated product clips. The following command records their local locations; it does not generate new footage:

```powershell
.venv\Scripts\python scripts/prepare_studio_demo.py --resources PATH_TO_RESOURCE_FOLDER --sample-id V005 --clips PATH_TO_MEDICUBE_CLIPS --music PATH_TO_PROJECT_MUSIC
```

The clips folder should contain `medicube-hero.mp4`, `medicube-texture.mp4`, `medicube-application_open.mp4` and `medicube-application_detail.mp4`. Omit `--music` for a silent export. The resulting `artifacts/demo/preset.json` remains local and is excluded from Git. On another computer, configure these paths again or upload reference/product materials directly.

## Live demonstration

1. Click **Try the Medicube example**, or upload a reference video of 1-90 seconds.
2. Click **Analyse reference**. The first run may take a few minutes to load the local models. Optional caption and beat analysis can be disabled for a quicker structural demonstration.
3. Show the detected shot count, pace, palette and timeline. Each slot has a function estimate, duration and editable caption. Opening/closing roles use position priors; other visual functions use CLIP prompt comparisons over three frames per shot. These scores are relative estimates, not calibrated confidence.
4. Review/correct the roles and captions. Caption position can be moved to the top or bottom. Upload product images or clips, and optionally tag their functions. A slot can use automatic matching or an explicitly selected material.
5. Enter the product name and optional user-provided talking points. Blank captions are filled with conservative product-specific copy; source captions are not copied into the reusable template.
6. Click **Create my video**. The renderer uses the new materials, optional replacement music, template durations and caption geometry. Open **Presentation view** for the side-by-side playback.
7. Download the result MP4, comparison MP4, template JSON and timing evidence. The template can be imported into a new session and filled with another material set without analysing the reference again.

For the meeting, open a completed saved session before presenting and keep its exported MP4 as a fallback. Refreshing the page restores the session. A background job is not resumed across a server restart; rerun analysis/render if interrupted. Existing completed media and saved templates remain available.

## What is automatic and what is limited

| Component | Current behaviour |
| --- | --- |
| Shot timing | Real adaptive PySceneDetect analysis using source presentation timestamps; projected to 24 export frames/s. |
| Shot functions | Local CLIP visual estimates plus opening/closing position rules. Human review is supported. No validated VLM semantics or zero/few-shot study yet. |
| Shot size | CLIP prompt estimates; known Phase 2 limitations still apply. |
| Caption geometry | Optional English EasyOCR sampling every 5 seconds; median placement estimate. No font identification or subtitle tracking. |
| Beats | Optional librosa HPSS candidates. No claimed accuracy or beat-synchronised rendering in this prototype. |
| Palette | Four dominant colours from representative frames. Used for the canvas; no exact colour-grading transfer. |
| Matching | Function tags and visual/shot-size scores with a reuse penalty; optional manual overrides. |
| Rendering | Hard cuts, new material timing, captions and replacement music. Short clips slow down up to 4x and then hold their final frame if needed; they do not loop inside a shot. The extension is recorded in evidence. Images are held as stills. |
| Validation | Exact output frame count, planned duration error and optional re-detected cut agreement. File identity checks are not proof of visual originality. |

The example clips are AI-generated illustrations. If a result/comparison slot lacks appropriate material, the export records a warning and uses an illustration; it must not be presented as a measured skin outcome. The original reference video/audio is not part of the new video. A separate comparison export intentionally includes the reference for the lab presentation.

This is one end-to-end prototype. The >=30-template clustered library, validated VLM semantics, 3 templates x 3 material sets, full beat alignment and Phase 5 originality evaluation remain open. Phase 2 beat/OCR/speech accuracy evaluation still awaits the reference annotations requested from the team.

## API for integration with the team's interface

All mutations require the header `X-Demo-Client: 1`. Media paths are opaque session URLs, not arbitrary filesystem paths. There are no cloud endpoints or authentication in this loopback prototype.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/sessions` | Create a local session. |
| `POST /api/sample` | Copy the configured sample into a fresh session. |
| `GET /api/sessions/{id}` | Read the public session/template/material state. |
| `POST /api/sessions/{id}/upload/reference` | Multipart field `files`, one video. |
| `POST /api/sessions/{id}/upload/assets` | Multipart field `files`, images/videos; max 20 materials. |
| `POST /api/sessions/{id}/upload/music` | Multipart field `files`, one audio file. |
| `PATCH /api/sessions/{id}/assets/{asset_id}` | JSON `{ "role": "texture_demonstration" }`; use `auto` for visual tagging. |
| `POST /api/sessions/{id}/analyse` | JSON `{ "read_text": true, "detect_beats": true }`; returns `job_id`. |
| `GET /api/jobs/{job_id}` | Poll stage, progress, status and any failure message. |
| `GET /api/sessions/{id}/template` | Download the reusable JSON. |
| `PUT /api/sessions/{id}/template` | Import/save JSON `{ "template": {...} }`. |
| `POST /api/sessions/{id}/render` | JSON `{ "template": {...}, "product_name": "...", "selling_points": [], "captions": true }`; returns `job_id`. |

Uploaded references are limited to 150 MB; individual materials/music to 60 MB. The template validates contiguous finite durations, function labels, caption boxes and bounded export sizes. Model work is serialised to avoid loading several copies on the small local machine. This architecture is ready for the Topic 1 team to call from their interface; broader deployment needs a separate server/hosting design.

Implementation uses the documented FastAPI [file upload](https://fastapi.tiangolo.com/tutorial/request-files/) and [static file](https://fastapi.tiangolo.com/tutorial/static-files/) interfaces.
