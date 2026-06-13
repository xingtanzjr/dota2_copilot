"""Minimap analysis: detect hero icons on the Dota 2 minimap.

Display modes
-------------
Dota 2 lets the player choose how heroes appear on the minimap:

* ``icons``  — hero portrait icons with a thin team-colored border  (P1: implemented)
* ``names``  — Chinese hero names colored by team                    (P2: TODO)
* ``arrows`` — per-hero colored arrows, ally/enemy shape differs     (P2: TODO)

The mode is picked from ``config/app.yaml -> minimap.display_mode``.

Algorithm for ``icons`` mode
----------------------------
Red/green pixels on the minimap come from four kinds of objects:

* hero icons   — 28-36 px square at 2K, only the BORDER is team-colored
                 (interior is the multicolor portrait)
* buildings    — 18-22 px square, ENTIRELY filled with team color
* couriers     — small filled circle (~10 px)
* creeps       — single/few pixels

Pipeline:
    1. Build raw HSV mask of red / green pixels.
    2. ``cv2.morphologyEx(..., MORPH_CLOSE)`` with a small kernel to fill the
       hero icon's border ring into a solid square.
    3. ``cv2.findContours`` on the closed mask, external only.
    4. Filter contours by bounding-box area (icon-sized) and aspect (square).
    5. **Hero vs. building discriminator**: recompute the fill ratio inside
       each bbox against the RAW (un-closed) mask.
           - low  fill (border only)   -> hero icon
           - high fill (solid block)   -> building, dropped

Couriers and creeps are filtered by the area lower bound.
Yellow camp markers fall outside both HSV ranges and are naturally ignored.

Known limitation (P1, acceptable)
---------------------------------
Two adjacent same-team icons get merged by morph-close into one elongated
blob; the aspect filter then rejects it. We may miss 1-2 heroes in stacked
fights, which is strictly safer than over-counting wards/buildings.
P2 template matching will recover identity and separate stacks.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import IconsDetectConfig, MinimapDetectConfig
from ..types import Frame, HeroBlob, Point, Team


# ---------------------------------------------------------------------------
# Color masking
# ---------------------------------------------------------------------------


def _build_color_mask(hsv: np.ndarray, h_ranges, s_min: int, v_min: int) -> np.ndarray:
    """Combine one or more H ranges (wrap-around safe) into a single binary mask."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_lo, h_hi in h_ranges:
        lower = np.array([h_lo, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_hi, 255, 255], dtype=np.uint8)
        mask |= cv2.inRange(hsv, lower, upper)
    return mask


# ---------------------------------------------------------------------------
# Icons-mode detection
# ---------------------------------------------------------------------------


def _detect_icons(
    raw_mask: np.ndarray,
    cfg: IconsDetectConfig,
) -> list[tuple[int, int, int, tuple[int, int, int, int]]]:
    """Find hero-icon contours in a team color mask.

    Returns ``[(cx, cy, contour_area, bbox), ...]`` where ``bbox`` is
    ``(x, y, w, h)`` in pixel coordinates inside the minimap crop.
    """
    if cfg.fill_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.fill_kernel, cfg.fill_kernel))
        closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k)
    else:
        closed_mask = raw_mask

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect_lo = 1.0 / cfg.aspect_tol
    aspect_hi = cfg.aspect_tol

    out: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        bbox_area = w * h
        if bbox_area < cfg.icon_area_min or bbox_area > cfg.icon_area_max:
            continue
        ratio = w / h
        if ratio < aspect_lo or ratio > aspect_hi:
            continue
        contour_area = int(cv2.contourArea(c))
        if contour_area <= 0:
            continue
        if contour_area / bbox_area < cfg.min_fill_ratio:
            continue

        # Hero vs. building: check fill against the RAW mask (no close).
        # A hero icon's interior is multicolor, so the raw red/green pixels
        # form just a ring -> low fill. A building is fully red/green -> high fill.
        raw_roi = raw_mask[y : y + h, x : x + w]
        raw_fill = float(np.count_nonzero(raw_roi)) / float(bbox_area)
        if raw_fill >= cfg.building_raw_fill_max:
            continue  # building, not a hero

        cx = x + w // 2
        cy = y + h // 2
        out.append((cx, cy, contour_area, (x, y, w, h)))
    return out


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class MinimapAnalyzer:
    """Stateless analyzer; instantiate once with config and reuse."""

    def __init__(self, cfg: MinimapDetectConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def detect(self, minimap_bgr: np.ndarray) -> tuple[list[HeroBlob], list[HeroBlob]]:
        """Run detection -> ``(enemies, allies)``.

        Raises ``NotImplementedError`` for display modes other than ``icons``
        until those detectors are built.
        """
        if minimap_bgr.ndim != 3 or minimap_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR image, got shape {minimap_bgr.shape}")

        if self.cfg.display_mode == "names":
            raise NotImplementedError(
                "Display mode 'names' requires OCR and is scheduled for P2. "
                "Switch Dota 2 minimap setting to 'icons' for now."
            )
        if self.cfg.display_mode == "arrows":
            raise NotImplementedError(
                "Display mode 'arrows' requires per-hero color/shape templates "
                "and is scheduled for P2. Switch to 'icons' for now."
            )

        h, w = minimap_bgr.shape[:2]
        hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)

        enemy_mask = _build_color_mask(
            hsv,
            self.cfg.enemy_red.h_ranges,
            self.cfg.enemy_red.s_min,
            self.cfg.enemy_red.v_min,
        )
        ally_mask = _build_color_mask(
            hsv,
            self.cfg.ally_green.h_ranges,
            self.cfg.ally_green.s_min,
            self.cfg.ally_green.v_min,
        )

        enemies_raw = _detect_icons(enemy_mask, self.cfg.icons)
        allies_raw = _detect_icons(ally_mask, self.cfg.icons)

        def to_blobs(raw, team: Team) -> list[HeroBlob]:
            return [
                HeroBlob(
                    team=team,
                    pos=Point(x=cx / w, y=cy / h),
                    pixel_pos=(cx, cy),
                    bbox=bbox,
                    area=area,
                )
                for cx, cy, area, bbox in raw
            ]

        return to_blobs(enemies_raw, Team.ENEMY), to_blobs(allies_raw, Team.ALLY)

    def analyze(self, minimap_bgr: np.ndarray, timestamp: float) -> Frame:
        """Convenience: detect + wrap into a ``Frame``."""
        enemies, allies = self.detect(minimap_bgr)
        h, w = minimap_bgr.shape[:2]
        return Frame(
            timestamp=timestamp,
            enemies=enemies,
            allies=allies,
            minimap_size=(w, h),
        )

    # ------------------------------------------------------------------
    # Debug helpers (used by tools/tune.py and debug_preview.py)
    # ------------------------------------------------------------------

    def debug_masks(self, minimap_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """Return intermediate masks for visual HSV-threshold tuning."""
        hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
        enemy_mask = _build_color_mask(
            hsv,
            self.cfg.enemy_red.h_ranges,
            self.cfg.enemy_red.s_min,
            self.cfg.enemy_red.v_min,
        )
        ally_mask = _build_color_mask(
            hsv,
            self.cfg.ally_green.h_ranges,
            self.cfg.ally_green.s_min,
            self.cfg.ally_green.v_min,
        )

        k = self.cfg.icons.fill_kernel
        if k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            enemy_closed = cv2.morphologyEx(enemy_mask, cv2.MORPH_CLOSE, kernel)
            ally_closed = cv2.morphologyEx(ally_mask, cv2.MORPH_CLOSE, kernel)
        else:
            enemy_closed = enemy_mask
            ally_closed = ally_mask

        return {
            "enemy_mask_raw": enemy_mask,
            "ally_mask_raw": ally_mask,
            "enemy_mask_closed": enemy_closed,
            "ally_mask_closed": ally_closed,
        }

