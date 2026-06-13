"""Live debug preview: continuously grab the minimap, run detection, overlay results.

Run via:  `dota2-copilot preview`

Keys (in the OpenCV window):
    q / ESC : quit
    s       : save the current annotated frame to `./snapshots/`
    d       : dump a full debug bundle (raw minimap + HSV masks + annotated)
              to `./snapshots/debug_<timestamp>/` for offline analysis
    m       : toggle mask-overlay mode (tint red/green where HSV matches)
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


WINDOW = "Dota 2 Copilot - Minimap Preview"

_TEAM_COLOR: dict[Team, tuple[int, int, int]] = {
    Team.ENEMY: (0, 0, 255),     # red box (BGR)
    Team.ALLY: (0, 255, 0),      # green box
    Team.UNKNOWN: (200, 200, 0),
}

_TEAM_LABEL: dict[Team, str] = {
    Team.ENEMY: "E",
    Team.ALLY: "A",
    Team.UNKNOWN: "?",
}


def _annotate(minimap_bgr: np.ndarray, frame: Frame) -> np.ndarray:
    img = minimap_bgr.copy()
    h, w = img.shape[:2]

    def draw(blob: HeroBlob) -> None:
        x, y, bw, bh = blob.bbox
        color = _TEAM_COLOR[blob.team]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
        cv2.circle(img, blob.pixel_pos, 4, color, 2)
        label = blob.hero_id[:8] if blob.hero_id else _TEAM_LABEL[blob.team]
        if blob.hero_id and blob.score:
            label = f"{blob.hero_id[:8]} {blob.score:.2f}"
        cv2.putText(
            img, label, (x, max(y - 2, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

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


def _overlay_masks(minimap_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Tint the minimap so user can SEE what HSV is matching.

    Red overlay = enemy mask hits, green overlay = ally mask hits.
    Uses the RAW (un-closed) masks so the user can judge HSV thresholds directly.
    """
    out = minimap_bgr.copy()
    enemy = masks["enemy_mask_raw"]
    ally = masks["ally_mask_raw"]

    overlay = np.zeros_like(out)
    overlay[enemy > 0] = (0, 0, 255)
    overlay[ally > 0] = (0, 255, 0)

    mask_any = (enemy > 0) | (ally > 0)
    blended = cv2.addWeighted(out, 0.35, overlay, 0.65, 0)
    out[mask_any] = blended[mask_any]
    return out


def _dump_debug(
    minimap_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    annotated: np.ndarray,
    out_root: Path,
    timestamp: float,
) -> Path:
    """Save raw minimap + all intermediate masks + annotated frame to disk."""
    ts = int(timestamp)
    out_dir = out_root / f"debug_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "minimap_raw.png"), minimap_bgr)
    cv2.imwrite(str(out_dir / "annotated.png"), annotated)
    for name, m in masks.items():
        cv2.imwrite(str(out_dir / f"{name}.png"), m)
    print(f"[preview] dumped debug bundle to {out_dir}")
    return out_dir


def run_preview(
    config: AppConfig | None = None,
    scale: float = 2.0,
    fps_override: float | None = None,
) -> None:
    """Run an interactive preview loop. `scale` enlarges the window for readability."""
    cfg = config or load_app_config()
    cal = load_minimap_calibration()
    analyzer = MinimapAnalyzer(cfg.minimap)

    effective_fps = fps_override if fps_override is not None else cfg.capture.fps
    interval = 1.0 / max(effective_fps, 0.1)
    snapshots_dir = Path("snapshots")

    print(
        f"[preview] minimap rect: {cal.minimap.width}x{cal.minimap.height} "
        f"at ({cal.minimap.x}, {cal.minimap.y}), fps={effective_fps}"
    )
    print("[preview] keys: q/ESC quit | s snapshot | d dump debug bundle | m toggle mask overlay")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    overlay_mode = False
    latest_minimap: np.ndarray | None = None
    latest_masks: dict[str, np.ndarray] | None = None
    latest_annotated: np.ndarray | None = None
    latest_ts = 0.0

    with open_grabber() as grabber:
        last_tick = 0.0
        while True:
            now = time.time()
            if now - last_tick < interval:
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    break
                continue
            last_tick = now

            minimap = grabber.grab(cal.rect())
            frame = analyzer.analyze(minimap, timestamp=now)
            masks = analyzer.debug_masks(minimap)
            annotated = _annotate(minimap, frame)

            latest_minimap = minimap
            latest_masks = masks
            latest_annotated = annotated
            latest_ts = now

            display = _overlay_masks(annotated, masks) if overlay_mode else annotated

            if scale != 1.0:
                display = cv2.resize(
                    display,
                    (int(display.shape[1] * scale), int(display.shape[0] * scale)),
                    interpolation=cv2.INTER_NEAREST,
                )

            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                snapshots_dir.mkdir(exist_ok=True)
                out = snapshots_dir / f"snapshot_{int(now)}.png"
                cv2.imwrite(str(out), display)
                print(f"[preview] saved {out}")
            if key == ord("d") and latest_minimap is not None and latest_masks is not None and latest_annotated is not None:
                _dump_debug(latest_minimap, latest_masks, latest_annotated, snapshots_dir, latest_ts)
            if key == ord("m"):
                overlay_mode = not overlay_mode
                print(f"[preview] mask overlay {'ON' if overlay_mode else 'OFF'}")

    cv2.destroyAllWindows()


def _countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    print(f"[dump] Switch to Dota now. Capturing in:")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)


def run_dump(
    config: AppConfig | None = None,
    delay: int = 3,
    from_image: Path | None = None,
    out_root: Path | None = None,
) -> Path:
    """Non-interactive: grab ONE frame, run detection, write full debug bundle, exit.

    Useful when the OpenCV preview window doesn't have keyboard focus and the
    interactive `m`/`d` keys can't be triggered.
    """
    cfg = config or load_app_config()
    cal = load_minimap_calibration()
    analyzer = MinimapAnalyzer(cfg.minimap)
    out_root = out_root or Path("snapshots")

    if from_image is not None:
        full = cv2.imread(str(from_image))
        if full is None:
            raise FileNotFoundError(f"Could not read image: {from_image}")
        r = cal.rect()
        h_img, w_img = full.shape[:2]
        if r.x + r.width > w_img or r.y + r.height > h_img:
            raise ValueError(
                f"Calibrated rect {r.width}x{r.height}@({r.x},{r.y}) "
                f"is outside image {w_img}x{h_img}. Recalibrate or pass a full-resolution screenshot."
            )
        minimap = full[r.y : r.y + r.height, r.x : r.x + r.width].copy()
        print(f"[dump] using image: {from_image}")
    else:
        _countdown(delay)
        with open_grabber() as grabber:
            minimap = grabber.grab(cal.rect())

    now = time.time()
    frame = analyzer.analyze(minimap, timestamp=now)
    masks = analyzer.debug_masks(minimap)
    annotated = _annotate(minimap, frame)
    overlay = _overlay_masks(annotated, masks)

    ts = int(now)
    out_dir = out_root / f"debug_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "minimap_raw.png"), minimap)
    cv2.imwrite(str(out_dir / "annotated.png"), annotated)
    cv2.imwrite(str(out_dir / "overlay_masks.png"), overlay)
    for name, m in masks.items():
        cv2.imwrite(str(out_dir / f"{name}.png"), m)

    print(
        f"[dump] minimap {minimap.shape[1]}x{minimap.shape[0]} "
        f"-> enemies={len(frame.enemies)} allies={len(frame.allies)}"
    )
    print(f"[dump] wrote bundle: {out_dir}")
    return out_dir


if __name__ == "__main__":
    run_preview()
