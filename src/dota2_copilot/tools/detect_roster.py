"""Auto-detect the 10 heroes in the top bar at the start of a Dota 2 match.

Top bar layout (left → right):
    | Radiant #1 | #2 | #3 | #4 | #5 |   clock   | Dire #1 | #2 | #3 | #4 | #5 |

This is meant to be called ONCE per match (in memory) by the live runtime,
since the 10 heroes change between games. A standalone CLI command exists too
for offline inspection.

Algorithm:
    1. Grab the top strip of the screen (top ~12% of height).
    2. For each of ~127 hero topbar templates (assets/topbar/<short>.png),
       at a few candidate widths, run cv2.matchTemplate.
    3. Keep the best score per hero, then NMS by horizontal distance to get
       up to 10 unique heroes.
    4. Sort by x position; left 5 = Radiant, right 5 = Dire.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..capture.screen import open_grabber
from ..config import (
    DEFAULT_ROSTER_PATH,
    REPO_ROOT,
    TopbarCalibration,
    hero_zh,
    load_topbar_calibration,
)
from ..types import ScreenRect


TOPBAR_DIR = REPO_ROOT / "assets" / "topbar"

# At 2560x1440 each top-bar hero portrait is roughly 70-110 px wide; at 1080p
# closer to 55-85 px. Try the union so a single tool works at any resolution.
CANDIDATE_WIDTHS: tuple[int, ...] = (55, 65, 75, 85, 95, 105, 115)

# matchTemplate score threshold for keeping a candidate; tuned empirically.
SCORE_THRESHOLD = 0.40

# Minimum horizontal distance between two roster slots (in pixels of the
# captured strip). 50 px easily covers the gap between adjacent slots without
# letting two detections collapse onto each other.
MIN_SLOT_DIST = 50


def _load_topbar_templates() -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    out: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for p in sorted(TOPBAR_DIR.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
        elif img.ndim == 3:
            bgr = img
            alpha = None
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alpha = None
        out.append((p.stem, bgr, alpha))
    return out


def _match_one(
    strip: np.ndarray,
    short: str,
    bgr: np.ndarray,
    alpha: np.ndarray | None,
    width: int,
) -> tuple[str, float, int, int, int, int] | None:
    """Return (short, score, x, y, w, h) or None if template doesn't fit."""
    h_t, w_t = bgr.shape[:2]
    target_h = int(round(h_t * width / w_t))
    H, W = strip.shape[:2]
    if target_h >= H or width >= W or target_h < 10 or width < 10:
        return None
    bgr_s = cv2.resize(bgr, (width, target_h), interpolation=cv2.INTER_AREA)
    if alpha is not None:
        alpha_s = cv2.resize(alpha, (width, target_h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(strip, bgr_s, cv2.TM_CCORR_NORMED, mask=alpha_s)
    else:
        res = cv2.matchTemplate(strip, bgr_s, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return short, float(score), int(loc[0]), int(loc[1]), width, target_h


def detect_roster_from_strip(
    strip_bgr: np.ndarray,
    slot_height: int | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Detect up to 10 unique heroes in a captured top-strip image.

    Returns a list (length <= 10) of dicts: {hero, score, x, y, w, h}.
    Sorted by x position (left -> right).

    If ``slot_height`` is provided (recommended; the height of the calibrated
    top-bar rect), template matching uses a single inferred width with a small
    +/-12% variance. This is far more accurate than the blind multi-scale sweep
    used when no calibration is available, because the top-bar HUD contains
    many distracting elements at random scales (HP bars, gold counters, kill
    score, etc.) that produce false positives at unrelated template sizes.
    """
    templates = _load_topbar_templates()
    if not templates:
        raise FileNotFoundError(
            f"No topbar templates in {TOPBAR_DIR}. Run scripts/fetch_assets.py."
        )

    if slot_height is not None and slot_height >= 20:
        # Template aspect is 16:9 (256x144). Slot width follows directly.
        slot_w = int(round(slot_height * 256 / 144))
        # +/- ~12% variance to handle any minor cropping.
        widths = (
            max(20, int(round(slot_w * 0.92))),
            slot_w,
            int(round(slot_w * 1.08)),
        )
        # With a tight crop the score floor can safely be lifted.
        score_thresh = 0.55
        min_dist = int(round(slot_w * 0.7))
    else:
        widths = CANDIDATE_WIDTHS
        score_thresh = SCORE_THRESHOLD
        min_dist = MIN_SLOT_DIST

    threads = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [
            pool.submit(_match_one, strip_bgr, short, bgr, alpha, w)
            for (short, bgr, alpha) in templates
            for w in widths
        ]
        results = [f.result() for f in futures]

    # Best (size, position) per hero.
    best: dict[str, tuple[str, float, int, int, int, int]] = {}
    for r in results:
        if r is None:
            continue
        short, score, *_ = r
        if short not in best or score > best[short][1]:
            best[short] = r

    # NMS by horizontal distance, keep top 10.
    sorted_dets = sorted(best.values(), key=lambda d: -d[1])
    kept: list[tuple[str, float, int, int, int, int]] = []
    for d in sorted_dets:
        if d[1] < score_thresh:
            continue
        if any(abs(d[2] - k[2]) < min_dist for k in kept):
            continue
        kept.append(d)
        if len(kept) >= max_results:
            break

    # Left -> right.
    kept.sort(key=lambda d: d[2])
    return [
        {"hero": d[0], "score": d[1], "x": d[2], "y": d[3], "w": d[4], "h": d[5]}
        for d in kept
    ]


@dataclass
class Roster:
    """In-memory roster discovered at runtime."""

    radiant: list[str] = field(default_factory=list)
    dire: list[str] = field(default_factory=list)
    my_team: str | None = None        # "radiant" | "dire" | None
    detections: list[dict] = field(default_factory=list)  # raw from detect_roster_from_strip

    @property
    def heroes(self) -> list[str]:
        return self.radiant + self.dire

    def allies(self) -> list[str]:
        if self.my_team == "radiant":
            return self.radiant
        if self.my_team == "dire":
            return self.dire
        return []

    def enemies(self) -> list[str]:
        if self.my_team == "radiant":
            return self.dire
        if self.my_team == "dire":
            return self.radiant
        return self.heroes  # team unknown -> treat all as enemies (safer)


def _countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    print("[roster] Switch to Dota now. Capturing in:")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)


def _detect_side(
    full: np.ndarray,
    rect: ScreenRect,
) -> list[dict]:
    """Crop ``full`` to ``rect``, detect up to 5 heroes, return full-screen-coord dicts."""
    H, W = full.shape[:2]
    x0 = max(0, min(rect.x, W - 1))
    y0 = max(0, min(rect.y, H - 1))
    x1 = max(x0 + 1, min(rect.x + rect.width, W))
    y1 = max(y0 + 1, min(rect.y + rect.height, H))
    strip = full[y0:y1, x0:x1].copy()
    slot_h = strip.shape[0]
    dets = detect_roster_from_strip(strip, slot_height=slot_h, max_results=5)
    for d in dets:
        d["x"] += x0
        d["y"] += y0
    return dets


def detect_roster_live(
    delay: int = 5,
    from_image: Path | None = None,
    my_team: str | None = None,
    strip_ratio: float = 0.12,
    topbar: TopbarCalibration | None = None,
) -> Roster:
    """Programmatic entry: grab screen -> detect 10 heroes -> return Roster.

    If a top-bar calibration is available (``config/topbar.json``), the two
    sides are processed independently with the clock/score area in the middle
    excluded -- this gives the cleanest possible match. Falls back to a blind
    full-strip scan only when no calibration exists.
    """
    if topbar is None:
        topbar = load_topbar_calibration()

    if from_image is not None:
        full = cv2.imread(str(from_image))
        if full is None:
            raise FileNotFoundError(f"Could not read {from_image}")
    else:
        _countdown(delay)
        with open_grabber() as g:
            full = g.grab_full()

    H, W = full.shape[:2]

    if topbar is not None:
        radiant_dets = _detect_side(full, topbar.radiant_rect())
        dire_dets = _detect_side(full, topbar.dire_rect())
        if not radiant_dets and not dire_dets:
            raise RuntimeError(
                "Detected 0 heroes on either side. Re-run "
                "`dota2-copilot calibrate-topbar` -- the saved rects probably "
                "don't match the current screen resolution."
            )
        return Roster(
            radiant=[d["hero"] for d in radiant_dets],
            dire=[d["hero"] for d in dire_dets],
            my_team=my_team,
            detections=radiant_dets + dire_dets,
        )

    # Fallback: blind full-strip scan (less accurate).
    strip_h = max(80, int(H * strip_ratio))
    strip = full[:strip_h, :].copy()
    dets = detect_roster_from_strip(strip, slot_height=None, max_results=10)
    if not dets:
        raise RuntimeError(
            "Detected 0 heroes in top bar. Hint: run "
            "`dota2-copilot calibrate-topbar` to mark the exact hero strips -- "
            "accuracy improves dramatically."
        )
    return Roster(
        radiant=[d["hero"] for d in dets[:5]],
        dire=[d["hero"] for d in dets[5:10]],
        my_team=my_team,
        detections=dets,
    )


def run_roster_detection(
    delay: int = 5,
    from_image: Path | None = None,
    out_path: Path | None = None,
    my_team: str | None = None,
    strip_ratio: float = 0.12,
    save_annotated: bool = True,
    save_to_disk: bool = True,
) -> Roster:
    """CLI-flavoured wrapper: detect, print summary, optionally persist to disk."""
    if from_image is None:
        print(f"[roster] capturing screen in {delay}s...")

    t0 = time.time()
    roster = detect_roster_live(
        delay=delay, from_image=from_image, my_team=my_team, strip_ratio=strip_ratio,
    )
    elapsed = time.time() - t0

    print(f"[roster] detected {len(roster.detections)} heroes in {elapsed:.1f}s\n")
    for i, d in enumerate(roster.detections):
        side = "天辉" if i < 5 else "夜魇"
        slot = (i % 5) + 1
        zh = hero_zh(d["hero"])
        print(
            f"  {side} #{slot}  {zh:<6} ({d['hero']:<20}) "
            f"score={d['score']:.3f}  x={d['x']}"
        )

    if save_to_disk:
        out_path = out_path or DEFAULT_ROSTER_PATH
        payload = {
            "heroes": roster.heroes,
            "radiant": roster.radiant,
            "dire": roster.dire,
            "my_team": roster.my_team,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[roster] saved snapshot -> {out_path} (debug only; runtime uses memory)")

    if save_annotated and from_image is None:
        # We don't have `strip` here in live mode anymore; skip annotated bundle.
        # (The CLI command captures its own bundle if needed.)
        pass

    return roster
