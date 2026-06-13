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

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..config import IconsDetectConfig, MinimapDetectConfig, REPO_ROOT, TemplateDetectConfig
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
# Template-matching detection (display_mode == "icons_template")
# ---------------------------------------------------------------------------


def _load_templates(
    cfg: TemplateDetectConfig,
    roster: set[str] | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """Load hero portrait templates at the configured size.

    If ``roster`` is given, only templates for those hero short-names are
    loaded (huge speed win once you know who's in the game).

    Returns ``[(short_name, bgr, alpha_or_None), ...]``.
    """
    tmpl_dir = Path(cfg.template_dir)
    if not tmpl_dir.is_absolute():
        tmpl_dir = REPO_ROOT / tmpl_dir

    out: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for p in sorted(tmpl_dir.glob(f"*_{cfg.template_size}.png")):
        short = p.stem.rsplit("_", 1)[0]
        if roster is not None and short not in roster:
            continue
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
        out.append((short, bgr, alpha))
    return out


def _detect_team_around(
    minimap_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    cfg: TemplateDetectConfig,
) -> Team:
    """Sample a ring just outside the bbox; majority red → enemy, green → ally."""
    x, y, w, h = bbox
    H, W = minimap_bgr.shape[:2]
    t = cfg.team_ring_thickness
    x0, y0 = max(x - t, 0), max(y - t, 0)
    x1, y1 = min(x + w + t, W), min(y + h + t, H)
    outer = minimap_bgr[y0:y1, x0:x1].copy()
    if outer.size == 0:
        return Team.UNKNOWN

    # Hollow out the icon interior — only the ring should contribute.
    cv2.rectangle(outer, (x - x0, y - y0), (x - x0 + w, y - y0 + h), (0, 0, 0), -1)
    hsv = cv2.cvtColor(outer, cv2.COLOR_BGR2HSV)

    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in cfg.team_red_h_ranges:
        red_mask |= cv2.inRange(
            hsv,
            np.array([lo, cfg.team_s_min, cfg.team_v_min], dtype=np.uint8),
            np.array([hi, 255, 255], dtype=np.uint8),
        )
    green_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in cfg.team_green_h_ranges:
        green_mask |= cv2.inRange(
            hsv,
            np.array([lo, cfg.team_s_min, cfg.team_v_min], dtype=np.uint8),
            np.array([hi, 255, 255], dtype=np.uint8),
        )

    red = int(np.count_nonzero(red_mask))
    green = int(np.count_nonzero(green_mask))
    if red < cfg.team_min_pixels and green < cfg.team_min_pixels:
        return Team.UNKNOWN
    return Team.ENEMY if red >= green else Team.ALLY


def _nms_by_center(detections: list[dict], dist: int) -> list[dict]:
    """Greedy non-max suppression by Euclidean distance between icon centers."""
    sorted_d = sorted(detections, key=lambda d: -d["score"])
    kept: list[dict] = []
    d2 = dist * dist
    for d in sorted_d:
        cx, cy = d["cx"], d["cy"]
        if any((cx - k["cx"]) ** 2 + (cy - k["cy"]) ** 2 < d2 for k in kept):
            continue
        kept.append(d)
    return kept


# OpenCV's matchTemplate releases the GIL, so a thread pool gives near-linear
# speedup. Keep a module-level pool so we don't recreate it every frame.
_TEMPLATE_THREADS = max(1, min(8, (os.cpu_count() or 4)))
_TEMPLATE_POOL: ThreadPoolExecutor | None = None


def _get_pool() -> ThreadPoolExecutor:
    global _TEMPLATE_POOL
    if _TEMPLATE_POOL is None:
        _TEMPLATE_POOL = ThreadPoolExecutor(max_workers=_TEMPLATE_THREADS)
    return _TEMPLATE_POOL


def _match_one(
    minimap_bgr: np.ndarray,
    short: str,
    bgr: np.ndarray,
    alpha: np.ndarray | None,
    threshold: float,
    size: int,
) -> dict | None:
    if alpha is not None:
        res = cv2.matchTemplate(minimap_bgr, bgr, cv2.TM_CCORR_NORMED, mask=alpha)
    else:
        res = cv2.matchTemplate(minimap_bgr, bgr, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < threshold:
        return None
    x, y = int(loc[0]), int(loc[1])
    return {
        "hero": short,
        "score": float(score),
        "x": x, "y": y,
        "cx": x + size // 2, "cy": y + size // 2,
        "bbox": (x, y, size, size),
    }


def _detect_templates(
    minimap_bgr: np.ndarray,
    templates: list[tuple[str, np.ndarray, np.ndarray | None]],
    cfg: TemplateDetectConfig,
) -> list[dict]:
    """Run all hero templates against the minimap; return surviving detections.

    Parallelized across templates with a thread pool (OpenCV releases the GIL).
    Each detection dict contains: hero, score, x, y, cx, cy, bbox (incl. size).
    """
    if not templates:
        return []
    size = cfg.template_size
    thresh = cfg.score_threshold
    pool = _get_pool()
    futures = [
        pool.submit(_match_one, minimap_bgr, short, bgr, alpha, thresh, size)
        for short, bgr, alpha in templates
    ]
    raw: list[dict] = [f for f in (fut.result() for fut in futures) if f is not None]
    return _nms_by_center(raw, cfg.nms_distance)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class MinimapAnalyzer:
    """Stateless analyzer; instantiate once with config and reuse.

    Match-time roster filtering: call ``set_roster(heroes, allies, enemies)``
    once at the start of a match to restrict template matching to the 10
    actually-picked heroes and to pre-decide team labels (no HSV fallback
    needed when the hero's side is known up-front).
    """

    def __init__(self, cfg: MinimapDetectConfig) -> None:
        self.cfg = cfg
        self._templates: list[tuple[str, np.ndarray, np.ndarray | None]] | None = None
        self._roster: set[str] | None = None
        # hero short -> Team.ALLY | Team.ENEMY (skip HSV ring sampling when known)
        self._team_by_hero: dict[str, Team] = {}

    # ------------------------------------------------------------------
    # Runtime roster injection
    # ------------------------------------------------------------------

    def set_roster(
        self,
        allies: list[str] | None = None,
        enemies: list[str] | None = None,
    ) -> None:
        """Restrict templates to the (allies + enemies) heroes and store sides.

        Pass ``None`` for both to clear and fall back to the full 127-hero pool.
        """
        if not allies and not enemies:
            self._roster = None
            self._team_by_hero = {}
        else:
            allies = allies or []
            enemies = enemies or []
            self._roster = set(allies) | set(enemies)
            self._team_by_hero = {
                **{h: Team.ALLY for h in allies},
                **{h: Team.ENEMY for h in enemies},
            }
        # Force template reload on next detect() call.
        self._templates = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def detect(self, minimap_bgr: np.ndarray) -> tuple[list[HeroBlob], list[HeroBlob]]:
        """Run detection -> ``(enemies, allies)``.

        Dispatches by ``display_mode``. Heroes whose team can't be determined
        are reported under whichever bucket the rules engine handles — we
        currently assign them to ``enemies`` so gank warnings stay safe.
        """
        if minimap_bgr.ndim != 3 or minimap_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR image, got shape {minimap_bgr.shape}")

        mode = self.cfg.display_mode
        if mode == "icons_template":
            return self._detect_via_templates(minimap_bgr)
        if mode == "icons":
            return self._detect_via_hsv(minimap_bgr)
        if mode == "names":
            raise NotImplementedError(
                "Display mode 'names' requires OCR and is scheduled for P2. "
                "Switch Dota 2 minimap setting to 'icons' for now."
            )
        if mode == "arrows":
            raise NotImplementedError(
                "Display mode 'arrows' requires per-hero color/shape templates "
                "and is scheduled for P2. Switch to 'icons' for now."
            )
        raise ValueError(f"Unknown display_mode: {mode!r}")

    # ------------------------------------------------------------------
    # display_mode == "icons"  (HSV color blob detection)
    # ------------------------------------------------------------------

    def _detect_via_hsv(self, minimap_bgr: np.ndarray) -> tuple[list[HeroBlob], list[HeroBlob]]:
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

    # ------------------------------------------------------------------
    # display_mode == "icons_template"  (template matching against assets/)
    # ------------------------------------------------------------------

    def _ensure_templates_loaded(self) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
        if self._templates is None:
            self._templates = _load_templates(self.cfg.template, roster=self._roster)
        return self._templates

    def _detect_via_templates(self, minimap_bgr: np.ndarray) -> tuple[list[HeroBlob], list[HeroBlob]]:
        h, w = minimap_bgr.shape[:2]
        templates = self._ensure_templates_loaded()
        dets = _detect_templates(minimap_bgr, templates, self.cfg.template)

        enemies: list[HeroBlob] = []
        allies: list[HeroBlob] = []
        for d in dets:
            bbox = d["bbox"]
            # If we know this hero's side from roster, trust it; otherwise
            # fall back to HSV ring sampling on the colored arrow.
            team = self._team_by_hero.get(d["hero"])
            if team is None:
                team = _detect_team_around(minimap_bgr, bbox, self.cfg.template)
            blob = HeroBlob(
                team=team,
                pos=Point(x=d["cx"] / w, y=d["cy"] / h),
                pixel_pos=(d["cx"], d["cy"]),
                bbox=bbox,
                area=bbox[2] * bbox[3],
                hero_id=d["hero"],
                score=d["score"],
            )
            # Unknown-team detections are bucketed with enemies so gank rules
            # err on the side of caution (false positive < missed danger).
            if team == Team.ALLY:
                allies.append(blob)
            else:
                enemies.append(blob)
        return enemies, allies

    # ------------------------------------------------------------------
    # Frame wrapper
    # ------------------------------------------------------------------

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

