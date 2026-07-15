> # ⚠️ DEPRECATED — moved into the CaptchaKraken monorepo
>
> This standalone repo is no longer maintained. Its detection + planning core
> now lives in the **`python/`** package of the unified monorepo:
> **https://github.com/JWriter20/CaptchaKraken** (PyPI: `captchakraken`).
> Use that repo for all new work; this one is archived read-only.

# CaptchaKraken CLI

The detection + planning core behind
[CaptchaKraken](https://github.com/JWriter20/CaptchaKraken). Given a
screenshot of a captcha challenge, it locates the image grid, asks a fine-tuned
**Qwen3.5-9B** vision LoRA which tiles to click, and emits the click plan the
browser solver replays.

> For the full solving-results showcase, demo videos, install flow, and
> self-hosting guide, see the parent repo
> **[PlaywrightCaptchaKrakenJS](https://github.com/JWriter20/PlaywrightCaptchaKrakenJS)**.
> This README is the technical reference for the CLI itself.

> ⭐ **Enjoying CaptchaKraken? Star both repos** and **watch** them for updates
> (smaller models, the hosted cloud API, new puzzle types):
> [CaptchaKraken-cli](https://github.com/JWriter20/CaptchaKraken-cli) (this engine)
> · [PlaywrightCaptchaKrakenJS](https://github.com/JWriter20/PlaywrightCaptchaKrakenJS)
> (the browser solver). On GitHub, use **Watch → All Activity** for release
> notifications.

## How it works (v2)

1. **`find_grid`** ([`src/tool_calls/find_grid.py`](src/tool_calls/find_grid.py)) —
   pure OpenCV, no model. Locates the 3×3 / 4×4 lattice by tracing
   consistent-colour separator lines, returning per-tile bounding boxes. This is
   the foundation of every solve and the thing CI guards most tightly.
2. **Grid planner** ([`src/planner.py`](src/planner.py)) — sends the numbered
   grid + prompt to the **Qwen3.5-9B grid LoRA** on a local **vLLM** server and
   parses the selected tile ids / boxes.
3. **Output** — the sequence of click actions, ready to replay in a browser
   automation stack (the Playwright wrapper drives this).

> **v1 note:** the old SAM3 + Docker-container detection flow is gone. v2 talks
> to a vLLM server only. (`Dockerfile` / `build_container.sh` remain for users
> who want to package their own server image, but the CLI no longer requires
> them.)

## Prerequisites

- A running **vLLM** server with the base model + grid LoRA loaded. The parent
  repo's [`install.sh`](https://github.com/JWriter20/PlaywrightCaptchaKrakenJS/blob/main/install.sh)
  sets this up and prints the exact `vllm serve …` command.
- **Python 3.10+**.

## Configuration

The CLI reads its endpoint and bearer from the environment (written by the parent
repo's `install.sh` into `captchakraken.env`):

| Variable | Meaning |
|---|---|
| `VLLM_BASE_URL` | Inference endpoint of your vLLM server (e.g. `http://localhost:8000/v1`). |
| `CAPTCHA_KRAKEN_API_KEY` | Bearer token for the server (`VLLM_API_KEY` is also accepted). |

## Usage

```bash
source ../captchakraken.env      # VLLM_BASE_URL + CAPTCHA_KRAKEN_API_KEY

# Solve a local image: classify → find_grid → plan. Prints the click actions
# (and an {"unsupported": true} error for non-grid puzzle types) as JSON.
python -m src.cli path/to/captcha.png

# Vendor hint (hCaptcha skips grid detection — find_grid false-positives on the
# header/footer bands of non-grid click puzzles):
python -m src.cli path/to/captcha.png --puzzle-source hcaptcha

# Retry hint after the vendor rejected an under-selection (e.g. reCAPTCHA's
# "Please select all matching images"):
python -m src.cli path/to/captcha.png --retry-mode missed-tiles
```

Optional positionals (kept for v1 argv compatibility): `<model>` overrides the
LoRA name registered with vLLM (default `captcha`); the bearer token can be
passed as the third positional instead of via the environment.

## Tests & CI

Grid detection is the CI guard — fast, deterministic, no GPU or network:

```bash
# Hermetic unit tests (run in CI on every PR)
python -m pytest tests/test_grid_detection_ci.py -q

# Full-corpus benchmark (report-only — prints per-type detection rates)
python -m pytest tests/test_find_grid_corpus.py -s
```

## License

Source-available under the **CaptchaKraken Source-Available License** — see
[LICENSE](LICENSE).
