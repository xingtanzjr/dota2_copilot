"""Record minimap frames + detection results to disk for offline replay / tuning.

Run via:  `dota2-copilot record --duration 120`

Outputs (under `recordings/<session_id>/`):
    frames/000001.png        — raw minimap crop (BGR, lossless)
    detections.jsonl         — one JSON line per frame (enemies/allies blob list)
    meta.json                — session metadata (resolution, fps, calibration)

The directory layout is intentionally simple so that later milestones (e.g. a
replay player for the rule engine) can iterate over `frames/*.png` and
`detections.jsonl` in lockstep.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from ..capture.minimap import MinimapAnalyzer
from ..capture.screen import open_grabber
from ..config import AppConfig, load_app_config, load_minimap_calibration
from ..types import Frame


def _frame_to_dict(frame: Frame, idx: int, image_relpath: str) -> dict:
    def blob_to_dict(b):
        return {
            "team": b.team.value,
            "pos": [b.pos.x, b.pos.y],
            "pixel_pos": list(b.pixel_pos),
            "bbox": list(b.bbox),
            "area": b.area,
        }

    return {
        "idx": idx,
        "timestamp": frame.timestamp,
        "image": image_relpath,
        "minimap_size": list(frame.minimap_size),
        "enemies": [blob_to_dict(b) for b in frame.enemies],
        "allies": [blob_to_dict(b) for b in frame.allies],
    }


def run_record(
    duration_seconds: float | None = None,
    session_name: str | None = None,
    config: AppConfig | None = None,
) -> Path:
    """Capture for `duration_seconds` (None = until Ctrl-C). Returns session dir."""
    cfg = config or load_app_config()
    cal = load_minimap_calibration()
    analyzer = MinimapAnalyzer(cfg.minimap)

    session_id = session_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = Path(cfg.capture.record.out_dir) / session_id
    frames_dir = base / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "session_id": session_id,
        "started_at": time.time(),
        "fps": cfg.capture.fps,
        "screen": {"width": cal.screen_width, "height": cal.screen_height},
        "minimap": {
            "x": cal.minimap.x,
            "y": cal.minimap.y,
            "width": cal.minimap.width,
            "height": cal.minimap.height,
        },
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    interval = 1.0 / max(cfg.capture.fps, 0.1)
    detections_path = base / "detections.jsonl"

    print(f"[record] session -> {base}")
    print(f"[record] fps={cfg.capture.fps}, duration={duration_seconds or 'unbounded'}")
    print("[record] press Ctrl-C to stop.")

    idx = 0
    started = time.time()
    try:
        with open_grabber() as grabber, detections_path.open("w", encoding="utf-8") as f_det:
            next_tick = started
            while True:
                if duration_seconds is not None and time.time() - started >= duration_seconds:
                    break

                now = time.time()
                if now < next_tick:
                    time.sleep(max(0.0, next_tick - now))
                    now = time.time()
                next_tick = now + interval

                minimap = grabber.grab(cal.rect())
                frame = analyzer.analyze(minimap, timestamp=now)

                rel = f"frames/{idx:06d}.png"
                cv2.imwrite(str(base / rel), minimap)
                f_det.write(json.dumps(_frame_to_dict(frame, idx, rel)) + "\n")
                f_det.flush()

                idx += 1
                if idx % 10 == 0:
                    print(
                        f"[record] {idx} frames, "
                        f"last: enemies={len(frame.enemies)} allies={len(frame.allies)}"
                    )
    except KeyboardInterrupt:
        print("\n[record] interrupted by user.")

    print(f"[record] done. {idx} frames written to {base}")
    return base


if __name__ == "__main__":
    run_record()
