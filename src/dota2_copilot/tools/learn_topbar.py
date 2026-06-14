"""Teach the roster detector new top-bar portrait variants (Arcana / Persona / etc).

Workflow
--------
1. You take a screenshot of a Dota match where one or more heroes shows up
   misidentified or unrecognized in ``dota2-copilot roster``.
2. You tell this tool which slot is actually which hero, e.g.::

       dota2-copilot learn-topbar --from-image scr.png \
           R3=pudge D1=phantom_assassin --label arcana

   The tool crops each named slot out of the screenshot using the saved
   top-bar calibration (``config/topbar.json``) and saves it to
   ``assets/topbar_variants/<hero>__<label>.png``.

3. Subsequent runs of ``dota2-copilot roster`` / ``preview`` automatically pick
   up the new variants -- they compete with the canonical portrait and the
   best-matching one wins per detection.

Slot codes
----------
* ``R1``..``R5``: Radiant slots, left to right.
* ``D1``..``D5``: Dire slots, left to right.

Each side's calibrated rect is split into 5 equal horizontal slots.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..capture.screen import open_grabber
from ..config import (
    REPO_ROOT,
    TopbarCalibration,
    hero_label,
    load_topbar_calibration,
    resolve_hero_short,
)
from ..types import ScreenRect

TOPBAR_VARIANTS_DIR = REPO_ROOT / "assets" / "topbar_variants"
_SLOT_RE = re.compile(r"^([RD])([1-5])$", re.IGNORECASE)


def _countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    print("[learn] Switch to Dota now. Capturing in:")
    for s in range(seconds, 0, -1):
        print(f"  {s}...", flush=True)
        time.sleep(1)


def _slot_rect(side_rect: ScreenRect, slot_index_0based: int) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) for slot N (0-based) inside ``side_rect``."""
    slot_w = side_rect.width / 5.0
    x = int(round(side_rect.x + slot_index_0based * slot_w))
    w = int(round(side_rect.x + (slot_index_0based + 1) * slot_w)) - x
    return (x, side_rect.y, w, side_rect.height)


def _parse_assignment(s: str) -> tuple[str, int, str]:
    """Parse ``R3=pudge`` or ``R3=小小`` -> ("R", 3, "<canonical short>").

    Accepts Chinese, English or short names on the right-hand side.
    Raises ValueError on bad input.
    """
    if "=" not in s:
        raise ValueError(f"expected 'R<1-5>=hero' or 'D<1-5>=hero', got: {s!r}")
    slot_str, hero = s.split("=", 1)
    m = _SLOT_RE.match(slot_str.strip())
    if not m:
        raise ValueError(f"bad slot code {slot_str!r}; use R1..R5 or D1..D5")
    side = m.group(1).upper()
    slot = int(m.group(2))
    raw = hero.strip()
    if not raw:
        raise ValueError(f"empty hero name in {s!r}")
    short = resolve_hero_short(raw)
    if short is None:
        raise ValueError(
            f"unknown hero {raw!r} (accepts Chinese name, English name, or short id, "
            "e.g. '小小' / 'Tiny' / 'tiny')"
        )
    return (side, slot, short)


def run_learn_topbar(
    assignments: list[str],
    from_image: Path | None = None,
    delay: int = 5,
    label: str = "variant",
    force: bool = False,
) -> int:
    """Crop and save labeled topbar variant portraits. Returns # files written."""
    # Validate label (filesystem-safe, no double underscore).
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")
    if not safe_label:
        print(f"[learn] invalid --label {label!r}", file=sys.stderr)
        return 0
    if "__" in safe_label:
        safe_label = safe_label.replace("__", "_")

    # Parse assignments.
    try:
        parsed = [_parse_assignment(a) for a in assignments]
    except ValueError as e:
        print(f"[learn] {e}", file=sys.stderr)
        return 0
    if not parsed:
        print("[learn] no slot=hero assignments given.", file=sys.stderr)
        return 0

    # Need topbar calibration.
    cal: TopbarCalibration | None = load_topbar_calibration()
    if cal is None:
        print(
            "[learn] No top-bar calibration. Run `dota2-copilot calibrate-topbar` first.",
            file=sys.stderr,
        )
        return 0

    # Capture image.
    if from_image is not None:
        full = cv2.imread(str(from_image))
        if full is None:
            print(f"[learn] could not read {from_image}", file=sys.stderr)
            return 0
        print(f"[learn] loaded {from_image} ({full.shape[1]}x{full.shape[0]})")
    else:
        _countdown(delay)
        with open_grabber() as g:
            full = g.grab_full()
        print(f"[learn] captured {full.shape[1]}x{full.shape[0]}")

    H, W = full.shape[:2]
    radiant_rect = cal.radiant_rect()
    dire_rect = cal.dire_rect()
    TOPBAR_VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for side, slot, hero in parsed:
        rect = radiant_rect if side == "R" else dire_rect
        x, y, w, h = _slot_rect(rect, slot - 1)
        x0 = max(0, min(x, W - 1))
        y0 = max(0, min(y, H - 1))
        x1 = max(x0 + 1, min(x + w, W))
        y1 = max(y0 + 1, min(y + h, H))
        crop = full[y0:y1, x0:x1].copy()
        if crop.size == 0:
            print(f"[learn] {side}{slot} crop is empty -- skipped.", file=sys.stderr)
            continue

        out_path = TOPBAR_VARIANTS_DIR / f"{hero}__{safe_label}.png"
        if out_path.exists() and not force:
            print(f"[learn] {out_path.name} already exists (use --force to overwrite); skipped.")
            continue
        cv2.imwrite(str(out_path), crop)
        print(
            f"[learn]  {side}{slot} -> {hero_label(hero)}  ({crop.shape[1]}x{crop.shape[0]})  "
            f"-> {out_path.relative_to(REPO_ROOT)}"
        )
        saved += 1

    print(f"[learn] saved {saved} variant template(s) to {TOPBAR_VARIANTS_DIR.relative_to(REPO_ROOT)}")
    return saved
