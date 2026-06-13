"""Throwaway experiment: template-match hero portraits + NMS + team detection.

Usage:  python scripts/experiment_match.py snapshots/debug_1781330566/minimap_raw.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "minimap"

# We tuned earlier that size=32 scale=1.0 gives the best scores at 2K minimap.
TEMPLATE_SIZE = 32

# Score threshold: matchTemplate response must clear this to count.
SCORE_THRESH = 0.55

# NMS: two detections whose icon centers are closer than this pixel distance
# are considered the same hero — keep the higher-scoring one.
NMS_DIST = 16

# Team detection: sample a ring AROUND the icon's bbox and count red vs green.
# The colored arrow/ring sits right at the icon edge in Dota's minimap.
TEAM_RING_THICKNESS = 7

# HSV ranges for the team-colored ring. Looser than the building HSV — the
# arrow can be dim against grass or other green/red terrain pixels.
RED_RANGES = [(0, 12), (165, 180)]
GREEN_RANGES = [(38, 90)]
S_MIN = 70
V_MIN = 70
# Minimum colored-pixel count to commit to a team (filters background noise).
MIN_TEAM_PIXELS = 6


def load_templates() -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    out = []
    for p in sorted(ASSETS.glob(f"*_{TEMPLATE_SIZE}.png")):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alpha = None
        short = p.stem.rsplit("_", 1)[0]
        out.append((short, bgr, alpha))
    return out


def match_one(minimap_bgr: np.ndarray, tmpl_bgr: np.ndarray, alpha: np.ndarray | None) -> tuple[float, tuple[int, int]]:
    if alpha is not None:
        res = cv2.matchTemplate(minimap_bgr, tmpl_bgr, cv2.TM_CCORR_NORMED, mask=alpha)
    else:
        res = cv2.matchTemplate(minimap_bgr, tmpl_bgr, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), max_loc


def detect_team(minimap_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, int, int]:
    """Sample a ring just outside the bbox and decide enemy/ally by HSV red vs green count."""
    x, y, w, h = bbox
    H, W = minimap_bgr.shape[:2]
    t = TEAM_RING_THICKNESS
    x0, y0 = max(x - t, 0), max(y - t, 0)
    x1, y1 = min(x + w + t, W), min(y + h + t, H)
    outer = minimap_bgr[y0:y1, x0:x1].copy()
    if outer.size == 0:
        return "unknown", 0, 0

    # Hollow out the inside of the icon (we only care about the colored ring).
    rel_x = x - x0
    rel_y = y - y0
    cv2.rectangle(outer, (rel_x, rel_y), (rel_x + w, rel_y + h), (0, 0, 0), -1)

    hsv = cv2.cvtColor(outer, cv2.COLOR_BGR2HSV)
    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in RED_RANGES:
        red_mask |= cv2.inRange(hsv, np.array([lo, S_MIN, V_MIN]), np.array([hi, 255, 255]))
    green_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in GREEN_RANGES:
        green_mask |= cv2.inRange(hsv, np.array([lo, S_MIN, V_MIN]), np.array([hi, 255, 255]))

    red = int(np.count_nonzero(red_mask))
    green = int(np.count_nonzero(green_mask))
    if red < MIN_TEAM_PIXELS and green < MIN_TEAM_PIXELS:
        return "unknown", red, green
    if red >= green:
        return "enemy", red, green
    return "ally", red, green


def nms(detections: list[dict], dist: int) -> list[dict]:
    """Greedy NMS by center distance."""
    sorted_d = sorted(detections, key=lambda d: -d["score"])
    kept: list[dict] = []
    for d in sorted_d:
        cx, cy = d["cx"], d["cy"]
        if any((cx - k["cx"]) ** 2 + (cy - k["cy"]) ** 2 < dist * dist for k in kept):
            continue
        kept.append(d)
    return kept


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <minimap.png>")
        sys.exit(1)

    minimap = cv2.imread(sys.argv[1])
    if minimap is None:
        sys.exit(f"could not read {sys.argv[1]}")
    print(f"minimap: {minimap.shape[1]}x{minimap.shape[0]}")

    templates = load_templates()
    print(f"loaded {len(templates)} templates @ {TEMPLATE_SIZE}px")

    t0 = time.time()
    raw_dets: list[dict] = []
    for short, bgr, alpha in templates:
        score, (x, y) = match_one(minimap, bgr, alpha)
        if score < SCORE_THRESH:
            continue
        bbox = (x, y, TEMPLATE_SIZE, TEMPLATE_SIZE)
        team, red, green = detect_team(minimap, bbox)
        raw_dets.append({
            "hero": short, "score": score, "team": team, "red": red, "green": green,
            "x": x, "y": y, "cx": x + TEMPLATE_SIZE // 2, "cy": y + TEMPLATE_SIZE // 2,
            "bbox": bbox,
        })
    elapsed = time.time() - t0
    print(f"matched in {elapsed:.2f}s, {len(raw_dets)} above threshold {SCORE_THRESH}")

    kept = nms(raw_dets, NMS_DIST)
    kept_unknown_filtered = [d for d in kept if d["team"] != "unknown"]
    print(f"after NMS: {len(kept)} unique  (of which team≠unknown: {len(kept_unknown_filtered)})")

    print(f"\n{'hero':<22} {'team':<7} {'score':>6} {'cx,cy':>10} {'red':>4} {'green':>5}")
    for d in sorted(kept, key=lambda r: -r["score"]):
        print(f"{d['hero']:<22} {d['team']:<7} {d['score']:>6.3f} {d['cx']:>4},{d['cy']:>4} {d['red']:>4} {d['green']:>5}")

    # Draw
    debug = minimap.copy()
    color_map = {"enemy": (0, 0, 255), "ally": (0, 255, 0), "unknown": (200, 200, 200)}
    for d in kept:
        c = color_map[d["team"]]
        x, y, w, h = d["bbox"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), c, 2)
        cv2.putText(debug, f"{d['hero'][:8]} {d['score']:.2f}",
                    (x, max(y - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)

    out_path = Path(sys.argv[1]).parent / "experiment_nms.png"
    cv2.imwrite(str(out_path), debug)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
