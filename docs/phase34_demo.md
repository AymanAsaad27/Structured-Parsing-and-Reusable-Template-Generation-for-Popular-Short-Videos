# First Phase 3/4 demonstration

One 20-second Medicube example has been rendered: five four-second slots, four unique Seedance 2.5 clips, captions and supplied project-use music. The hero clip is reused at the end. Output is 720x1280, 24 fps, 480 frames, AAC 48 kHz. This is a manual reference-informed prototype, not the completed 30-template library or an automatically extracted template.

V001's annotations were grouped into hook, product/method, core action, visual evidence and closing. The source has 25 shots over about 58 seconds. The demonstration condenses those groups and replaces the result-first opening with an application illustration. Full traceability is in examples/templates/medicube_demonstration_20s_v1.json.

## Reproduction

Install `python -m pip install -e ".[generation]"`. Save the Ark key only in local `.env` as ARK_API_KEY. It is excluded from Git. Model service/billing must be enabled independently.

The Seedance helper requires an explicit `--submit` for a new paid task. Each run name keeps its task ID and prevents blind resubmission after ambiguous network failures. To resume a queued/running task, omit `--submit`. Inspect existing failed attempts before starting another run.

```sh
python scripts/seedance_pilot.py --run-name medicube-application_open --request-file examples/generation/medicube/application_open.request.json --submit
python scripts/seedance_pilot.py --run-name medicube-application_open
```

Run names for the first demo: medicube-hero, medicube-application_open, medicube-texture and medicube-application_detail. The example request files contain public product reference URLs and prompts, not credentials. First generation used 720p, 9:16, duration 4 and generate_audio false. The generated texture and use scenes are illustrative, not measured product performance.

Create a local asset JSON mapping each template run_name to an MP4 path. Relative paths resolve from that JSON's directory. Then:

```sh
python scripts/render_template.py --template examples/templates/medicube_demonstration_20s_v1.json --assets artifacts/assets.json --music PATH_TO_PERMITTED_MUSIC --font PATH_TO_TTF --output artifacts/final.mp4 --work-dir artifacts/render
```

The renderer validates contiguous slots, checks clip duration, trims, normalizes resolution/FPS, overlays captions, concatenates and mixes music with fade and a -22 LUFS normalization target. Fonts and music must be supplied locally. The first example used M02 Original Soft Pulse, whose supplied manifest permits project use.

## Findings

- Re-parsing detected all four planned cuts at frames 96, 192, 288 and 384 within +/-2 frames. This checks the edit, not independent video-understanding accuracy.
- Tube shape and main logo are recognizable in sampled frames. Fine label text is not verified.
- The tool changes between flat and metal spatulas; continuity needs improvement.
- The music produced no detected beat grid. Fixed-duration cuts are not claimed as verified beat-synced editing.
- Usage is 349,200 tokens for four unique clips, including the earlier hero; three new clips use 261,900 tokens. This is not an invoice.

Only one product/material set and one template video are completed. VLM zero/few-shot experiments, the >=30-template library, clustering, automated material matching and the full 3x3 examples remain future work. Media, raw API status URLs and credentials stay under ignored local artifacts; aggregate evidence is under reports/phase34/.
