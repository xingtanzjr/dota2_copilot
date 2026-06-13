"""Live debug preview: continuously grab the minimap, run detection, overlay results.

Run via:  `dota2-copilot preview`

Keys (in the OpenCV window):
    q / ESC : quit
    s       : save the current annotated frame to `./snapshots/`
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..capture.minimap import MinimapAnalyzer
from ..capture.screen import open_grabber
from ..config import AppConfig, load_app_config, load_minimap_calibration
from ..types import Frame, HeroBlob, Team


WINDOW = "Dota 2 Copilot — Minimap Preview"

_TEAM_COLOR: dict[Team, tuple[int, int, int]] = {
    Team.ENEMY: (0, 0, 255),     # red box (BGR)
    Team.ALLY: (0, 255, 0),      # green box
    Team.UNKNOWN: (200, 200, 0),
}


def _annotate(minimap_bgr: np.ndarray, frame: Frame) -> np.ndarray:
    img = minimap_bgr.copy()
    h, w = img.shape[:2]

    def draw(blob: HeroBlob) -> None:
        x, y, bw, bh = blob.bbox
        color = _TEAM_COLOR[blob.team]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 1)
        cv2.circle(img, blob.pixel_pos, 2, color, -1)

    for b in frame.enemies:
        draw(b)
    for b in frame.allies:
        draw(b)

    header = f"enemies={len(frame.enemies)}  allies={len(frame.allies)}  {w}x{h}"
    cv2.putText(
        img, header, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return img


def run_preview(config: AppConfig | None = None, scale: float = 2.0) -> None:
    """Run an interactive preview loop. `scale` enlarges the window for readability."""
    cfg = config or load_app_config()
    cal = load_minimap_calibration()
    analyzer = MinimapAnalyzer(cfg.minimap)

    interval = 1.0 / max(cfg.capture.fps, 0.1)
    snapshots_dir = Path("snapshots")

    print(
        f"[preview] minimap rect: {cal.minimap.width}x{cal.minimap.height} "
        f"at ({cal.minimap.x}, {cal.minimap.y}), fps={cfg.capture.fps}"
    )
    print("[preview] press 'q' / ESC to quit, 's' to save snapshot.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    with open_grabber() as grabber:
        last_tick = 0.0
        while True:
            now = time.time()
            if now - last_tick < interval:
                # Don't spin: yield a bit, but stay responsive to keys.
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    break
                continue
            last_tick = now

            minimap = grabber.grab(cal.rect())
            frame = analyzer.analyze(minimap, timestamp=now)
            annotated = _annotate(minimap, frame)

            if scale != 1.0:
                annotated = cv2.resize(
                    annotated,
                    (int(annotated.shape[1] * scale), int(annotated.shape[0] * scale)),
                    interpolation=cv2.INTER_NEAREST,
                )

            cv2.imshow(WINDOW, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                snapshots_dir.mkdir(exist_ok=True)
                out = snapshots_dir / f"snapshot_{int(now)}.png"
                cv2.imwrite(str(out), annotated)
                print(f"[preview] saved {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_preview()
