# Dota 2 Copilot

A minimap-aware Dota 2 assistant. It watches the part of the screen you can
already see (the minimap), tracks hero positions over a sliding window, and
fires rule-based reminders (gank warning, push safety, lane safety, etc.).

> Compliance: this tool reads pixels from your own screen only — no memory
> reads, no injection, no input automation. Safe with respect to VAC.

See [design.md](docs/design.md) for the full architecture.

---

## Status

**Milestone 1 (current)** — end-to-end "I can see the heroes":

- [x] Project skeleton + config loader
- [x] Screen grab via `mss`
- [x] HSV color-segmentation hero detector
- [x] Interactive minimap calibration tool
- [x] Live debug preview window
- [x] Frame-by-frame recording for offline replay

Later milestones (state store, rule engine, Windows Toast, LLM, etc.) are
described in [docs/design.md](docs/design.md).

---

## Install

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e .
```

Python 3.11+ required.

> **Note:** Calibration and preview need a real display (cannot run headless).
> Run them on the gaming machine where Dota 2 is launched.

---

## First-time setup (one-time, on the gaming machine)

1. Start Dota 2, get into a match (or a demo / replay — anything that shows
   the minimap on screen). The minimap should be in its **default left-bottom**
   position.

2. Open a second window / Alt-Tab and run:

   ```bash
   dota2-copilot calibrate
   ```

   A full-screen screenshot opens. **Drag a rectangle around the minimap**,
   then press ENTER (or SPACE). The selection is saved to
   `config/minimap.json`.

   For 2K (2560×1440) the minimap is roughly `x=0..340, y=1100..1440`. The
   tighter you select, the fewer false positives later.

---

## Daily use

### Live debug preview

```bash
dota2-copilot preview
```

Opens an OpenCV window showing the minimap crop with:
- **Red boxes** around detected enemies
- **Green boxes** around detected allies / yourself
- Header text with detection counts

Keys: `q` / `ESC` quit, `s` save annotated snapshot to `./snapshots/`.

### Record a session for offline analysis

```bash
dota2-copilot record --duration 600     # 10-minute capture
# or just:
dota2-copilot record                    # unbounded; Ctrl-C to stop
```

Writes to `recordings/<timestamp>/`:
- `frames/000001.png …` — raw minimap crops
- `detections.jsonl`    — one JSON per frame with all blobs
- `meta.json`           — session metadata

These recordings will drive offline tuning and (later) replay-based rule
engine evaluation without needing a live game.

---

## Configuration

Edit `config/app.yaml` to tweak:
- `capture.fps` — sampling rate (default 1 Hz)
- `capture.history_seconds` — sliding window length (used from M2 onward)
- `minimap.enemy_red` / `minimap.ally_green` — HSV thresholds (raise `s_min` /
  `v_min` if you see false positives from background terrain)
- `minimap.blob_area_min` / `blob_area_max` — icon-size filter

Changes are picked up on the next CLI invocation.

---

## Troubleshooting

**"Minimap calibration not found"** — run `dota2-copilot calibrate` first.

**Detection misses heroes / picks up creeps**
  - Tighten `blob_area_min`/`max` in `config/app.yaml` (icons are larger than
    creep dots).
  - Raise `s_min`/`v_min` to reject washed-out colors.
  - Re-run calibration with a tighter minimap rectangle.

**Window appears off-screen / can't see preview**
  - The OpenCV window is created with `WINDOW_NORMAL` and is resizable;
    drag it from your taskbar preview.

---

## Project layout

```
dota2_copilot/
├── pyproject.toml
├── config/
│   ├── app.yaml          # defaults, versioned
│   └── minimap.json      # produced by `calibrate`, gitignored
├── docs/design.md
├── src/dota2_copilot/
│   ├── cli.py            # typer entry point
│   ├── config.py         # pydantic config loaders
│   ├── types.py          # HeroBlob, Frame, Point, Team, …
│   ├── capture/
│   │   ├── screen.py     # mss wrapper
│   │   └── minimap.py    # HSV segmentation + blob detection
│   └── tools/
│       ├── calibrate_minimap.py
│       ├── debug_preview.py
│       └── record.py
└── recordings/           # produced by `record`, gitignored
```
