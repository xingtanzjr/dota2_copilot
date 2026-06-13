"""Interactive map-landmark calibration.

Run via: ``dota2-copilot calibrate-landmarks [--from-image PATH]``

Walks through every landmark defined in ``assets/map_landmarks.yaml`` and
prompts the user to left-click its position on the minimap. The refined
coordinates are saved to ``config/map_landmarks.json`` (user-specific,
gitignored). The shipped YAML template is never modified.

Inputs
------
* No ``--from-image``: grab the live screen, then crop to the calibrated
  minimap rect from ``config/minimap.json`` (requires ``calibrate`` first).
* ``--from-image PATH``: load an image file. If its size matches the
  calibrated minimap rect it's used as-is; otherwise it's treated as a full
  screenshot and cropped using the calibration. If no minimap calibration
  exists, the whole image is used as the minimap.

Keys
----
* Left-click  set current landmark
* n           next (skip; keep existing value)
* p           previous (go back one landmark)
* u           undo the click for the current landmark
* s           save & exit
* q / Esc     quit without saving
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from ..capture.screen import open_grabber
from ..config import (
    DEFAULT_LANDMARKS_OVERRIDE,
    load_map_landmarks,
    load_minimap_calibration,
    save_map_landmarks_override,
)


WINDOW = "Dota 2 Copilot - Calibrate Landmarks"

# UI colors (BGR).
_C_OTHER_SET = (90, 200, 90)     # green   — other landmarks already set
_C_OTHER_UNSET = (90, 90, 90)    # grey    — other landmarks not yet set
_C_CURRENT = (0, 200, 255)       # yellow  — currently editing
_C_CURRENT_OLD = (0, 120, 200)   # orange  — current's previous position
_C_TEXT = (255, 255, 255)


def _load_minimap_image(from_image: Path | None) -> np.ndarray:
    """Return a BGR numpy image of the minimap region."""
    if from_image is not None:
        img = cv2.imread(str(from_image))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {from_image}")
        try:
            cal = load_minimap_calibration()
            mw, mh = cal.minimap.width, cal.minimap.height
            ih, iw = img.shape[:2]
            if (iw, ih) == (mw, mh):
                return img  # already cropped to minimap
            if iw >= cal.minimap.x + mw and ih >= cal.minimap.y + mh:
                return img[
                    cal.minimap.y : cal.minimap.y + mh,
                    cal.minimap.x : cal.minimap.x + mw,
                ]
        except FileNotFoundError:
            pass  # no calibration — use whole image
        return img

    # Live-screen mode.
    cal = load_minimap_calibration()
    with open_grabber() as grabber:
        return grabber.grab(cal.rect())


def _draw(
    canvas: np.ndarray,
    landmarks: dict[str, tuple[float, float]],
    set_keys: set[str],
    cur_name: str,
    cur_orig: tuple[float, float] | None,
    scale: int,
) -> np.ndarray:
    """Render the minimap with all landmark markers + current-edit highlight."""
    h, w = canvas.shape[:2]
    disp = cv2.resize(canvas, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Draw all landmarks (except the current) as small filled circles.
    for name, (nx, ny) in landmarks.items():
        if name == cur_name:
            continue
        px = int(round(nx * w * scale))
        py = int(round(ny * h * scale))
        col = _C_OTHER_SET if name in set_keys else _C_OTHER_UNSET
        cv2.circle(disp, (px, py), 4, col, -1, lineType=cv2.LINE_AA)

    # Current landmark's previous position (if it was already set).
    if cur_orig is not None:
        px = int(round(cur_orig[0] * w * scale))
        py = int(round(cur_orig[1] * h * scale))
        cv2.drawMarker(
            disp, (px, py), _C_CURRENT_OLD, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA
        )

    # Current landmark's new position (a big bright circle).
    nx, ny = landmarks[cur_name]
    px = int(round(nx * w * scale))
    py = int(round(ny * h * scale))
    cv2.circle(disp, (px, py), 10, _C_CURRENT, 2, lineType=cv2.LINE_AA)
    cv2.drawMarker(disp, (px, py), _C_CURRENT, cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
    return disp


def _put_status(
    img: np.ndarray,
    idx: int,
    total: int,
    name: str,
    has_set: bool,
) -> None:
    """Draw a status bar at the top of the image."""
    bar_h = 60
    cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (30, 30, 30), -1)
    line1 = f"[{idx + 1}/{total}]  {name}"
    line2 = "L-click: set   n: next   p: prev   u: undo   s: save+exit   q/Esc: quit"
    cv2.putText(img, line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_TEXT, 1, cv2.LINE_AA)
    if has_set:
        cv2.putText(
            img, "modified", (img.shape[1] - 120, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _C_CURRENT, 1, cv2.LINE_AA,
        )


def run_landmark_calibration(
    from_image: Path | None = None,
    scale: int = 3,
    out_path: Path | None = None,
) -> Path | None:
    """Interactive landmark calibration. Returns the saved file path, or None."""
    minimap_img = _load_minimap_image(from_image)
    mh, mw = minimap_img.shape[:2]
    print(f"[calibrate-landmarks] minimap image: {mw}x{mh}")

    base = load_map_landmarks()
    names: list[str] = base.names()
    if not names:
        print("[calibrate-landmarks] no landmarks defined in template.", file=sys.stderr)
        return None
    print(f"[calibrate-landmarks] {len(names)} landmarks to review")

    # Working copy of normalized coords (starts from template + any saved override).
    coords: dict[str, tuple[float, float]] = dict(base.landmarks)
    originals: dict[str, tuple[float, float]] = dict(coords)
    modified: set[str] = set()

    state = {"click_xy": None}

    def on_mouse(event, x, y, flags, param):  # noqa: ARG001
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click_xy"] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    idx = 0
    save_flag = False
    while 0 <= idx < len(names):
        name = names[idx]
        # Render current state.
        disp = _draw(
            minimap_img,
            coords,
            modified,
            name,
            originals.get(name) if name in modified else None,
            scale,
        )
        # Status bar is drawn into an extended canvas.
        status_h = 60
        framed = np.zeros((disp.shape[0] + status_h, disp.shape[1], 3), dtype=np.uint8)
        framed[status_h:, :, :] = disp
        _put_status(framed, idx, len(names), name, name in modified)
        cv2.imshow(WINDOW, framed)

        key = cv2.waitKey(20) & 0xFF
        if state["click_xy"] is not None:
            cx, cy = state["click_xy"]
            state["click_xy"] = None
            # Translate display coords back to normalized minimap coords.
            # Display = top status bar (status_h) + scaled minimap.
            mx = cx / scale
            my = (cy - status_h) / scale
            if 0 <= my <= mh and 0 <= mx <= mw:
                coords[name] = (round(mx / mw, 4), round(my / mh, 4))
                modified.add(name)
                idx += 1  # auto-advance after a click
            continue

        if key in (ord("q"), 27):  # q / Esc
            print("[calibrate-landmarks] aborted (no changes saved).")
            break
        if key == ord("s"):
            save_flag = True
            break
        if key == ord("n"):
            idx += 1
        elif key == ord("p"):
            idx = max(0, idx - 1)
        elif key == ord("u"):
            if name in modified:
                coords[name] = originals[name]
                modified.remove(name)

    cv2.destroyWindow(WINDOW)

    if not save_flag and idx < len(names):
        return None

    # If we walked off the end without explicitly hitting 's', still save.
    if not modified:
        print("[calibrate-landmarks] nothing modified — not writing override.")
        return None

    saved = save_map_landmarks_override(coords, out_path)
    print(f"[calibrate-landmarks] modified {len(modified)} landmarks -> {saved}")
    return saved


if __name__ == "__main__":
    run_landmark_calibration()
