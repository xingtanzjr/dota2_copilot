"""Produce a *clean* minimap background (no heroes/units) for synthetic data.

The Dota minimap terrain is static, so a single clean crop is enough as a
compositing base. The *ideal* source is a real in-game capture with no units on
screen (e.g. the first second of a bot/demo match, or a spectator view before
creeps spawn). If you have one, you don't need this script -- just drop it into
your backgrounds folder.

This script is the fallback when all you have is a normal capture with units on
it: it masks out unit-like pixels (team-coloured buildings/creeps/arrows, bright
portraits, camp/ward markers) and inpaints the terrain underneath.

    python scripts/make_clean_background.py snapshots/debug_1781330566/minimap_raw.png \
        -o assets/minimap_bg/clean_day.png --debug

Quality note: inpainting is approximate. Prefer a real clean capture when you
can get one; use --debug to eyeball the mask and result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_unit_mask(bgr: np.ndarray, sat_thresh: int, val_hi: int) -> np.ndarray:
    """Mask pixels that belong to units/markers rather than terrain.

    Terrain on the minimap is low-saturation blue-grey/green-brown. Units are
    either strongly team-coloured or high-saturation portraits, so a saturation
    threshold plus explicit red/green/blue/yellow masks captures most of them.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # High-saturation (portraits, bright markers) but ignore near-black noise.
    sat = (s >= sat_thresh) & (v >= 40)

    # Explicit strong team colours (arrows, buildings, creeps).
    b, g, r = bgr[..., 0].astype(int), bgr[..., 1].astype(int), bgr[..., 2].astype(int)
    red = (r > 150) & (r - g > 55) & (r - b > 55)
    green = (g > 130) & (g - r > 45) & (g - b > 30)
    blue = (b > 140) & (b - r > 45) & (b - g > 10)
    yellow = (r > 160) & (g > 150) & (b < 120)
    bright = v >= val_hi

    mask = (sat | red | green | blue | yellow | bright).astype(np.uint8) * 255

    # Consolidate into blobs and grow to cover icon edges / anti-aliasing.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    return mask


def make_clean(bgr: np.ndarray, sat_thresh: int, val_hi: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    mask = build_unit_mask(bgr, sat_thresh, val_hi)
    clean = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    # A second, gentler pass smooths seams where big blobs were removed.
    mask2 = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    clean = cv2.inpaint(clean, mask2, max(2, radius // 2), cv2.INPAINT_NS)
    return clean, mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="minimap crop with units on it")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output clean background PNG")
    ap.add_argument("--sat-thresh", type=int, default=90, help="saturation above this = unit")
    ap.add_argument("--val-hi", type=int, default=225, help="value above this = unit highlight")
    ap.add_argument("--radius", type=int, default=4, help="inpaint radius (px)")
    ap.add_argument("--debug", action="store_true", help="also write *_mask.png and side-by-side")
    args = ap.parse_args()

    bgr = cv2.imread(str(args.src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read {args.src}")

    clean, mask = make_clean(bgr, args.sat_thresh, args.val_hi, args.radius)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), clean)
    print(f"wrote {args.out}  ({clean.shape[1]}x{clean.shape[0]})")

    if args.debug:
        mp = args.out.with_name(args.out.stem + "_mask.png")
        sp = args.out.with_name(args.out.stem + "_sidebyside.png")
        cv2.imwrite(str(mp), mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(sp), np.hstack([bgr, mask_bgr, clean]))
        print(f"wrote {mp} and {sp}")


if __name__ == "__main__":
    main()
