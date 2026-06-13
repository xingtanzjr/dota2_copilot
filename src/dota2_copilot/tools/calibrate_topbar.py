"""Interactive calibration: select the top-bar hero strips on screen.

Run via:  ``dota2-copilot calibrate-topbar``
   or:    ``dota2-copilot calibrate-topbar --from-image path/to/screenshot.png``
   or:    ``dota2-copilot calibrate-topbar --delay 5``

You drag **two** rectangles in succession:

    1. The left strip containing the 5 Radiant hero portraits.
    2. The right strip containing the 5 Dire hero portraits.

The clock / score area in the middle is skipped entirely. Vertically, include
**only** the portrait area (skip HP/gold bars below).

Result -> ``config/topbar.json`` (consumed by ``detect_roster_live``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..capture.screen import open_grabber
from ..config import TopbarCalibration, save_topbar_calibration
from ..types import ScreenRect

# ASCII title so Windows OpenCV doesn't render mojibake.
WINDOW = "Dota 2 Copilot - Calibrate Top Bar"

DISPLAY_MAX_W = 1800
DISPLAY_MAX_H = 1000


def _countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    print("[calibrate-topbar] Switch to Dota now. Capturing in:")
    for s in range(seconds, 0, -1):
        print(f"  {s}...", flush=True)
        time.sleep(1)


def _select_rect_interactive(
    image: np.ndarray,
    title: str,
    subtitle: str,
) -> tuple[int, int, int, int] | None:
    h_img, w_img = image.shape[:2]
    scale = min(DISPLAY_MAX_W / w_img, DISPLAY_MAX_H / h_img, 1.0)
    if scale < 1.0:
        disp_w = int(round(w_img * scale))
        disp_h = int(round(h_img * scale))
        display_base = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        disp_w, disp_h = w_img, h_img
        display_base = image.copy()

    state: dict = {"start": None, "end": None, "dragging": False, "aborted": False}

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

    bar_h = 70
    confirmed = False
    while True:
        canvas = display_base.copy()
        if state["start"] is not None and state["end"] is not None:
            cv2.rectangle(canvas, state["start"], state["end"], (0, 255, 255), 2)
        cv2.rectangle(canvas, (0, 0), (disp_w, bar_h), (30, 30, 30), -1)
        cv2.putText(
            canvas, title,
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, subtitle,
            (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "ENTER/SPACE: confirm   R: reset   ESC: abort",
            (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):
            confirmed = True
            break
        if key == 27:
            state["aborted"] = True
            break
        if key == ord("r"):
            state["start"] = None
            state["end"] = None
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                state["aborted"] = True
                break
        except cv2.error:
            state["aborted"] = True
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if state["aborted"] or not confirmed:
        return None
    if state["start"] is None or state["end"] is None:
        return None

    x1, y1 = state["start"]
    x2, y2 = state["end"]
    if x1 == x2 or y1 == y2:
        return None

    x = int(round(min(x1, x2) / scale))
    y = int(round(min(y1, y2) / scale))
    w = int(round(abs(x2 - x1) / scale))
    h = int(round(abs(y2 - y1) / scale))
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    return (x, y, w, h)


def run_topbar_calibration(
    out_path: Path | None = None,
    from_image: Path | None = None,
    delay: int = 3,
) -> Path | None:
    if from_image is not None:
        screenshot = cv2.imread(str(from_image))
        if screenshot is None:
            print(f"[calibrate-topbar] Could not read image: {from_image}", file=sys.stderr)
            return None
        screen_h, screen_w = screenshot.shape[:2]
        print(f"[calibrate-topbar] Loaded image {screen_w}x{screen_h}")
    else:
        _countdown(delay)
        with open_grabber() as grabber:
            screen_w, screen_h = grabber.primary_size()
            screenshot = grabber.grab_full()
        print(f"[calibrate-topbar] Captured primary monitor: {screen_w}x{screen_h}")

    # Step 1: Radiant strip.
    print("[calibrate-topbar] Step 1/2: drag a rectangle around the 5 RADIANT portraits (left).")
    radiant = _select_rect_interactive(
        screenshot,
        title="Step 1/2: drag around the 5 RADIANT portraits (LEFT side).",
        subtitle="Skip the clock + scores in the middle. Skip HP/gold bars below.",
    )
    if radiant is None:
        print("[calibrate-topbar] Aborted at step 1.", file=sys.stderr)
        return None

    # Step 2: Dire strip.
    print("[calibrate-topbar] Step 2/2: drag a rectangle around the 5 DIRE portraits (right).")
    dire = _select_rect_interactive(
        screenshot,
        title="Step 2/2: drag around the 5 DIRE portraits (RIGHT side).",
        subtitle="Skip the clock + scores in the middle. Skip HP/gold bars below.",
    )
    if dire is None:
        print("[calibrate-topbar] Aborted at step 2.", file=sys.stderr)
        return None

    rx, ry, rw, rh = radiant
    dx, dy, dw, dh = dire
    radiant_rect = ScreenRect(x=rx, y=ry, width=rw, height=rh)
    dire_rect = ScreenRect(x=dx, y=dy, width=dw, height=dh)
    cal = TopbarCalibration.from_rects(screen_w, screen_h, radiant_rect, dire_rect)
    saved = save_topbar_calibration(cal, out_path)

    inferred_slot_w_r = int(round(rh * 256 / 144))
    inferred_slot_w_d = int(round(dh * 256 / 144))
    print(
        f"[calibrate-topbar] RADIANT: {rw}x{rh} at ({rx},{ry})  ~slot {inferred_slot_w_r}x{rh}"
    )
    print(
        f"[calibrate-topbar] DIRE:    {dw}x{dh} at ({dx},{dy})  ~slot {inferred_slot_w_d}x{dh}"
    )
    print(f"[calibrate-topbar] Saved -> {saved}")
    return saved
