# Beauty Video Templates

Structured parsing and reusable editing templates for 15–60 second vertical beauty videos.

**Status: Phase 2 implemented with expanded experiments; three module accuracy evaluations remain open.** Training now uses the 200-video collection with a frozen 140/20/40 split. Reviewed beat timestamps, OCR boxes/text and speech transcripts still need to be located for their evaluation. A manually adapted template has also produced one 20-second Medicube demonstration; automatic semantic extraction and clustering remain planned. See [Phase 2 usage](docs/phase2_usage.md), [expanded results](reports/phase2/improvements/RESULTS.md) and [first generation example](docs/phase34_demo.md).

On 30 new shot-test videos, content/adaptive boundary F1 is **0.840 / 0.869**, within ±2 source frames; adaptive is now the default. On 629 reference-conditioned test shots, CLIP zero-shot/probe accuracy is **32.6% / 59.8%**. The four-class transition CNN scores **87.1% accuracy**, but dissolve recall is only **3.8%**. Supplementary grouped size CV confirms poor medium/wide recall, so Phase 2 is not signed off. Full supports, confusion matrices and limits are in the [new comparison](reports/phase2/improvements/module_comparison.csv). The [original pilot report](reports/phase2/Phase_2_Report.html) is retained separately.

## Quick start

Use Python 3.11 or 3.12 for model environments. The lightweight core also supports newer Python; model packages may have narrower compatibility.

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
bvt doctor
bvt validate examples/results/shots.json
python -m pytest
ruff check .
ruff format --check .
```

FFmpeg is installed separately. Install optional model dependencies only when needed, for example `python -m pip install -e ".[shots]"`. No models download during core installation.

## Repository layout

```text
.github/                  CI, issue templates, pull request checklist
configs/phase2/           Per-module experiment configuration drafts
src/beauty_video_templates/
  parsing/               Shots, beats, OCR, ASR, transitions, shot size
  semantics/             Shot function labeling (Phase 3)
  templates/             Abstraction and clustering (Phase 3)
  matching/              Material assignment (Phase 4)
  rendering/             Timeline assembly and synthesis (Phase 4)
  evaluation/            Module and system metrics (Phases 2/5)
  pipelines/             Future orchestration
  schemas/               Packaged parsing-result schema
  contracts.py           Shared validation and backend interface
  cli.py                 Working doctor and validate commands
data/                    Raw/interim/processed data, metadata, annotations, splits
models/                  Provenance conventions; weights stay local
examples/                Small synthetic examples
experiments/             Experiment protocol and aggregate results
reports/                 Deliverables and Phase 2 comparison outline
prompts/                 Versioned semantic labeling prompt
notebooks/               Exploration conventions
apps/                    Future demonstration interface
scripts/                 Environment snapshot utility
tests/                   Contract and CLI behavior tests
artifacts/               Local run outputs, excluded from Git
docs/                    Setup, architecture, contracts, roadmap, publishing
```

## Research roadmap

1. Research and proposal.
2. Dataset collection and annotation.
3. **Phase 2: independent parsing modules and baseline comparisons.**
4. Semantic analysis, reusable templates and clustering.
5. Material matching and video rendering.
6. Structural/content similarity and user evaluation.
7. Integrated demonstration and reproducibility report.

Start with [Phase 2 tasks](docs/phase2.md), [setup](docs/setup.md), and [architecture](docs/architecture.md).

## Data and licensing

Do not commit original videos, private media, weights or credentials. Store permitted metadata and video IDs. Record data permissions and model licenses independently. Similarity scores are not proof of originality.

No open-source code license has been selected automatically; see [licensing](docs/licensing.md). Read [contribution guidance](CONTRIBUTING.md) and [publishing instructions](docs/publishing.md).
