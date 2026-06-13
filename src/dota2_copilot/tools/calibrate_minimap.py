"""Interactive calibration: select the minimap region on screen.

Run via:  `dota2-copilot calibrate`

The tool takes a full-screen screenshot, opens it in an OpenCV window, and
asks the user to drag a rectangle around the minimap. The result is written
to `config/minimap.json` and reused by every subsequent command.

For 2K (2560x1440) the minimap defaults to roughly:
    x=0..340, y=1100..1440
but the exact pixel rect depends on HUD scale, ultra-wide, etc., so we don't
hard-code it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from ..capture.screen import open_grabber
from ..config import MinimapCalibration, save_minimap_calibration
from ..types import ScreenRect


WINDOW = "Dota 2 Copilot — Calibrate Minimap"


def run_calibration(out_path: Path | None = None) -> Path | None:
    """Capture screen, prompt user to drag a rectangle, save the result."""
    with open_grabber() as grabber:
        screen_w, screen_h = grabber.primary_size()
        screenshot = grabber.grab_full()

    print(f"[calibrate] Primary monitor: {screen_w}x{screen_h}")
    print("[calibrate] Drag a rectangle around the minimap.")
    print("[calibrate] Press ENTER or SPACE to confirm, 'c' to cancel.")

    # selectROI handles its own window lifecycle.
    roi = cv2.selectROI(WINDOW, screenshot, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(WINDOW)

    x, y, w, h = (int(v) for v in roi)
    if w == 0 or h == 0:
        print("[calibrate] Empty selection — aborted.", file=sys.stderr)
        return None

    rect = ScreenRect(x=x, y=y, width=w, height=h)
    cal = MinimapCalibration.from_rect(screen_w, screen_h, rect)
    saved = save_minimap_calibration(cal, out_path)

    print(
        f"[calibrate] Saved minimap region {rect.width}x{rect.height} "
        f"at ({rect.x}, {rect.y}) -> {saved}"
    )
    return saved


if __name__ == "__main__":
    run_calibration()
