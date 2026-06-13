"""Interactive calibration: select the minimap region on screen.

Run via:  `dota2-copilot calibrate`
   or:    `dota2-copilot calibrate --from-image path/to/screenshot.png`
   or:    `dota2-copilot calibrate --delay 5`

The tool either:
* (default) waits ``--delay`` seconds with a countdown so you can Alt+Tab back
  to Dota, then takes a full-screen screenshot, or
* (``--from-image``) loads a screenshot file you prepared with Win+Shift+S.

Then it opens an interactive window where you drag a rectangle around the
minimap. The result is written to ``config/minimap.json``.

Keys
----
* Left-drag   draw a rectangle
* ENTER/SPACE confirm
* R           reset (clear the rectangle)
* ESC         abort without saving

The window can also be closed via the X button — that also aborts cleanly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..capture.screen import open_grabber
from ..config import MinimapCalibration, save_minimap_calibration
from ..types import ScreenRect


# ASCII-only so Windows OpenCV doesn't render mojibake in the title bar.
WINDOW = "Dota 2 Copilot - Calibrate Minimap"

# If the screenshot is larger than these, the picker will downscale just for
# display (and rescale clicks back to original-image coords).
DISPLAY_MAX_W = 1600
DISPLAY_MAX_H = 900


def _countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    print("[calibrate] Switch to Dota now. Capturing in:")
    for s in range(seconds, 0, -1):
        print(f"  {s}...", flush=True)
        time.sleep(1)


def _select_rect_interactive(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Drag-to-select rectangle, returns (x, y, w, h) in original-image coords.

    Returns ``None`` if the user aborts (ESC / window close / empty rect).
    """
    h_img, w_img = image.shape[:2]

    # Scale down for display if the screenshot is bigger than the window cap.
    scale = min(DISPLAY_MAX_W / w_img, DISPLAY_MAX_H / h_img, 1.0)
    if scale < 1.0:
        disp_w = int(round(w_img * scale))
        disp_h = int(round(h_img * scale))
        display_base = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        disp_w, disp_h = w_img, h_img
        display_base = image.copy()

    state: dict[str, object] = {
        "start": None,      # (x, y) in display coords
        "end": None,        # (x, y) in display coords
        "dragging": False,
        "aborted": False,
    }

    def on_mouse(event, x, y, flags, param):  # noqa: ARG001
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["end"] = (x, y)
            state["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["end"] = (x, y)
            state["dragging"] = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, disp_w, disp_h)
    cv2.setMouseCallback(WINDOW, on_mouse)

    bar_h = 50
    confirmed = False
    while True:
        canvas = display_base.copy()
        # Current selection rectangle (yellow).
        if state["start"] is not None and state["end"] is not None:
            cv2.rectangle(canvas, state["start"], state["end"], (0, 255, 255), 2)
        # Instruction bar.
        cv2.rectangle(canvas, (0, 0), (disp_w, bar_h), (30, 30, 30), -1)
        cv2.putText(
            canvas, "Drag a rectangle around the minimap.",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "ENTER/SPACE: confirm   R: reset   ESC: abort",
            (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # ENTER or SPACE
            confirmed = True
            break
        if key == 27:  # ESC
            state["aborted"] = True
            break
        if key == ord("r"):
            state["start"] = None
            state["end"] = None

        # User closed the window via the X button — treat as abort.
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                state["aborted"] = True
                break
        except cv2.error:
            state["aborted"] = True
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)  # let HighGUI flush the destroy event

    if state["aborted"] or not confirmed:
        return None
    if state["start"] is None or state["end"] is None:
        return None

    x1, y1 = state["start"]
    x2, y2 = state["end"]
    if x1 == x2 or y1 == y2:
        return None

    # Translate display coords back to original-image coords.
    x = int(round(min(x1, x2) / scale))
    y = int(round(min(y1, y2) / scale))
    w = int(round(abs(x2 - x1) / scale))
    h = int(round(abs(y2 - y1) / scale))
    # Clamp to image bounds.
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    return (x, y, w, h)


def run_calibration(
    out_path: Path | None = None,
    from_image: Path | None = None,
    delay: int = 3,
) -> Path | None:
    """Capture screen (or load an image), pick a rectangle, save the result."""
    if from_image is not None:
        screenshot = cv2.imread(str(from_image))
        if screenshot is None:
            print(f"[calibrate] Could not read image: {from_image}", file=sys.stderr)
            return None
        screen_h, screen_w = screenshot.shape[:2]
        print(f"[calibrate] Loaded image {screen_w}x{screen_h} from {from_image}")
    else:
        _countdown(delay)
        with open_grabber() as grabber:
            screen_w, screen_h = grabber.primary_size()
            screenshot = grabber.grab_full()
        print(f"[calibrate] Captured primary monitor: {screen_w}x{screen_h}")

    print("[calibrate] Drag a rectangle around the minimap. ENTER to confirm.")

    result = _select_rect_interactive(screenshot)
    if result is None:
        print("[calibrate] Aborted (no calibration saved).", file=sys.stderr)
        return None

    x, y, w, h = result
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
