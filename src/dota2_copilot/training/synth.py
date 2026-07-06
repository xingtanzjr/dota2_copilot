"""Synthetic minimap frame generator for YOLO training.

Why synthetic?
--------------
We already own every hero's minimap *portrait* (``assets/minimap/<short>_32.png``,
portrait only -- the game draws the team-coloured border at render time). By
compositing those portraits -- with a synthesized team border + directional
arrow -- onto real minimap backgrounds, we can mint thousands of perfectly
labelled frames for free, instead of hand-annotating screenshots.

What one synthetic frame contains
---------------------------------
1. A real minimap background crop, lightly augmented (resize / HSV jitter /
   blur), so the model sees varied terrain lighting.
2. **Negatives come from the real background itself**: the two clean captures
   already contain the game's real buildings, creep waves, camps and couriers
   at authentic positions/colours/sizes, so the detector learns to ignore them
   for free. (Synthetic distractors are available but OFF by default -- the
   real buildings are static, so fabricated ones add unrealistic noise rather
   than useful hard negatives.)
3. A realistic **5v5 roster**: 10 DISTINCT heroes, 5 ally + 5 enemy. Each is
   only *visible* with some probability -- allies drop out when dead/respawning,
   enemies drop out under fog of war (much more often) -- so a frame carries
   anywhere from ~1 up to 10 icons, matching real games. Each visible hero =
   portrait + directional team arrow (blue = ally / red = enemy), randomly
   placed (may overlap for teamfight realism), with a YOLO label
   ``(class_id, cx, cy, w, h)`` in normalized coordinates.

Team is encoded ONLY by the hero's arrow colour (blue=ally, red=enemy), never
by map position or building colour: your base may sit bottom-left or top-right
(so the green/red building corners can swap). We cover both by supplying two
real background captures (green-BL and green-TR); the backgrounds are used
as-is (never rotated -- rotation would produce unnatural terrain).

Everything here needs only numpy + opencv (no torch, no pydantic), so it runs
in a minimal environment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .classes import class_id, load_hero_shorts

# training/ -> dota2_copilot/ -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_DIR = REPO_ROOT / "assets" / "minimap"


# ---------------------------------------------------------------------------
# Colours (BGR), sampled per-object so the model never latches onto one exact
# shade. Measured from real 2K minimap captures.
#
# IMPORTANT asymmetry seen on the real minimap:
#   * Hero DIRECTION ARROW: ally = BLUE, enemy = RED
#   * Buildings / creeps  : ally = GREEN, enemy = RED
# The hero portrait itself has NO team-coloured border -- only the little
# arrow encodes team. So the arrow is the single discriminative team cue.
# ---------------------------------------------------------------------------

# Hero directional-arrow colours.
_ALLY_ARROW_LO = np.array([200, 110, 10], dtype=np.int16)   # bright blue/cyan
_ALLY_ARROW_HI = np.array([255, 215, 60], dtype=np.int16)
_ENEMY_ARROW_LO = np.array([10, 20, 190], dtype=np.int16)   # bright red
_ENEMY_ARROW_HI = np.array([60, 75, 255], dtype=np.int16)

# Building / creep colours (distractors only).
_ALLY_UNIT_LO = np.array([30, 170, 30], dtype=np.int16)     # green
_ALLY_UNIT_HI = np.array([110, 255, 110], dtype=np.int16)
_ENEMY_UNIT_LO = np.array([30, 30, 170], dtype=np.int16)    # red
_ENEMY_UNIT_HI = np.array([80, 80, 255], dtype=np.int16)

# Neutral camp / ward markers.
_YELLOW_BGR = (40, 200, 235)


def _sample_between(lo: np.ndarray, hi: np.ndarray, rng: random.Random) -> tuple[int, int, int]:
    c = [int(rng.randint(int(lo[i]), int(hi[i]))) for i in range(3)]
    return (c[0], c[1], c[2])


def _sample_arrow_color(team: str, rng: random.Random) -> tuple[int, int, int]:
    """Hero direction-arrow colour: ally = blue, enemy = red."""
    lo, hi = (_ALLY_ARROW_LO, _ALLY_ARROW_HI) if team == "ally" else (_ENEMY_ARROW_LO, _ENEMY_ARROW_HI)
    return _sample_between(lo, hi, rng)


def _sample_unit_color(team: str, rng: random.Random) -> tuple[int, int, int]:
    """Building / creep colour: ally = green, enemy = red."""
    lo, hi = (_ALLY_UNIT_LO, _ALLY_UNIT_HI) if team == "ally" else (_ENEMY_UNIT_LO, _ENEMY_UNIT_HI)
    return _sample_between(lo, hi, rng)


# ---------------------------------------------------------------------------
# Portrait bank
# ---------------------------------------------------------------------------


@dataclass
class PortraitBank:
    """Loads and caches hero portraits (BGR + optional alpha) at base size."""

    template_dir: Path = DEFAULT_TEMPLATE_DIR
    base_size: int = 32
    _by_short: dict[str, tuple[np.ndarray, np.ndarray | None]] = field(default_factory=dict)

    def load(self, shorts: list[str]) -> "PortraitBank":
        for short in shorts:
            p = self.template_dir / f"{short}_{self.base_size}.png"
            if not p.exists():
                continue
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] == 4:
                bgr = img[:, :, :3].copy()
                alpha = img[:, :, 3].copy()
            elif img.ndim == 3:
                bgr = img
                alpha = _silhouette_alpha(bgr)  # black-bg template -> real shape
            else:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                alpha = _silhouette_alpha(bgr)
            self._by_short[short] = (bgr, alpha)
        if not self._by_short:
            raise FileNotFoundError(
                f"No hero portraits found under {self.template_dir} "
                f"(expected *_{self.base_size}.png). Run scripts/fetch_assets.py."
            )
        return self

    def available(self) -> list[str]:
        return list(self._by_short.keys())

    def get(self, short: str) -> tuple[np.ndarray, np.ndarray | None]:
        return self._by_short[short]


# ---------------------------------------------------------------------------
# Icon rendering (portrait + border + arrow)
# ---------------------------------------------------------------------------


def _rounded_square_mask(size: int, radius: int) -> np.ndarray:
    """Filled rounded-square alpha mask (uint8 0/255)."""
    m = np.zeros((size, size), dtype=np.uint8)
    r = max(0, min(radius, size // 2))
    cv2.rectangle(m, (r, 0), (size - r, size), 255, -1)
    cv2.rectangle(m, (0, r), (size, size - r), 255, -1)
    for cx, cy in ((r, r), (size - r, r), (r, size - r), (size - r, size - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def _silhouette_alpha(bgr: np.ndarray) -> np.ndarray:
    """Recover the portrait's real shape from a black-background template.

    The ``*_NN.png`` templates are RGB with a *baked-in black background* (the
    circular hero portrait sits on solid black). Compositing that black would
    punch an opaque hole in the minimap, so we rebuild the true alpha: keep the
    non-black portrait silhouette (filled, no interior holes), drop the black.
    """
    lum = bgr.max(2).astype(np.uint8)
    m = (lum > 14).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m
    fill = np.zeros_like(m)
    cv2.drawContours(fill, cnts, -1, 255, -1)  # solid -> removes interior holes
    return fill


def render_hero_icon(
    portrait_bgr: np.ndarray,
    portrait_alpha: np.ndarray | None,
    size: int,
    team: str,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Render a minimap hero icon = circular portrait + team direction arc.

    Matches the real Dota minimap: the portrait has **no** team-coloured border;
    team is shown by a soft **curved gradient arc** hugging the portrait's edge
    on one side (blue = ally, red = enemy), pointing in the facing/move
    direction -- a glowing crescent, not a hard triangle.

    Returns ``(icon_bgr, icon_alpha, pad)``. The canvas is ``(size + 2*pad)``
    square with the portrait centred, so the caller can place the portrait
    square at ``(x, y)`` by pasting the canvas at ``(x - pad, y - pad)``. The
    label box is exactly the portrait square (arc excluded → centre stays on
    the true hero position).
    """
    pad = int(np.ceil(size * 0.36)) + 3
    canvas = size + 2 * pad
    bgr = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    alpha = np.zeros((canvas, canvas), dtype=np.uint8)

    ox = oy = pad  # portrait square top-left inside canvas
    center = pad + size / 2.0

    # --- portrait (real silhouette from the black-bg template, no border) ---
    radius = int(size * rng.uniform(0.10, 0.20))
    sq_mask = _rounded_square_mask(size, radius)
    port = cv2.resize(portrait_bgr, (size, size), interpolation=cv2.INTER_AREA)
    pa = None
    if portrait_alpha is not None:
        pa = cv2.resize(portrait_alpha, (size, size), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.35:  # tiny rotation for variety (rotate portrait + alpha together)
        ang = rng.uniform(-6, 6)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
        port = cv2.warpAffine(port, M, (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
        if pa is not None:
            pa = cv2.warpAffine(pa, M, (size, size), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    port_alpha = sq_mask.copy()
    if pa is not None:
        port_alpha = cv2.min(port_alpha, pa)
    roi = bgr[oy:oy + size, ox:ox + size]
    a3 = (port_alpha.astype(np.float32) / 255.0)[..., None]
    bgr[oy:oy + size, ox:ox + size] = (port.astype(np.float32) * a3 + roi * (1 - a3)).astype(np.uint8)
    alpha[oy:oy + size, ox:ox + size] = cv2.max(alpha[oy:oy + size, ox:ox + size], port_alpha)

    # --- directional team indicator: soft curved gradient arc (blue = ally,
    #     red = enemy) hugging the portrait edge; a glowing crescent that fades
    #     out toward its ends and radially, NOT a mechanical triangle. ---
    if rng.random() < 0.95:  # a few frames have no arc (just-appeared/idle)
        color = np.array(_sample_arrow_color(team, rng), dtype=np.float32)
        theta = rng.uniform(0, 2 * np.pi)          # facing / move direction
        R = size / 2.0
        ys, xs = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
        dx = xs - center
        dy = ys - center
        r = np.sqrt(dx * dx + dy * dy)
        ang = np.arctan2(dy, dx)
        da = np.arctan2(np.sin(ang - theta), np.cos(ang - theta))  # signed angular dist
        half = np.radians(rng.uniform(62, 88))     # crescent angular half-width (wider)
        r_peak = R * rng.uniform(1.04, 1.14)       # sits clearly outside the edge
        r_sigma = R * rng.uniform(0.17, 0.24)      # thicker radial band
        radial = np.exp(-0.5 * ((r - r_peak) / r_sigma) ** 2)
        angw = np.where(np.abs(da) <= half,
                        0.5 * (1.0 + np.cos(np.pi * (da / half))),  # cosine taper -> gradient ends
                        0.0)
        glow = (radial * angw).astype(np.float32)
        mx = float(glow.max())
        if mx > 1e-6:
            glow = glow / mx
            glow = glow ** 0.55                    # gamma: fuller, bolder crescent (not a thin peak)
            glow *= rng.uniform(0.95, 1.0)         # near-opaque core so the arc reads clearly
        g3 = glow[..., None]
        bgr[:] = np.clip(bgr.astype(np.float32) * (1 - g3) + color * g3, 0, 255).astype(np.uint8)
        alpha[:] = np.maximum(alpha, (glow * 255).astype(np.uint8))

    # --- per-icon photometric jitter ---
    if rng.random() < 0.6:
        gain = rng.uniform(0.85, 1.15)
        bias = rng.uniform(-12, 12)
        jbgr = np.clip(bgr.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        bgr = np.where(alpha[..., None] > 0, jbgr, bgr)
    if rng.random() < 0.3:
        noise = np.random.normal(0, rng.uniform(2, 8), bgr.shape)
        jbgr = np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        bgr = np.where(alpha[..., None] > 0, jbgr, bgr)

    return bgr, alpha, pad


def _paste_rgba(dst: np.ndarray, icon_bgr: np.ndarray, icon_alpha: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite ``icon`` onto ``dst`` with top-left at ``(x, y)`` (clipped)."""
    H, W = dst.shape[:2]
    ih, iw = icon_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + iw), min(H, y + ih)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    a = (icon_alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[..., None]
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    src = icon_bgr[sy0:sy1, sx0:sx1].astype(np.float32)
    dst[y0:y1, x0:x1] = (src * a + roi * (1 - a)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Distractors (drawn, never labelled)
# ---------------------------------------------------------------------------


def _draw_distractors(canvas: np.ndarray, rng: random.Random) -> None:
    """Draw buildings / creeps / couriers / markers that must be IGNORED."""
    H, W = canvas.shape[:2]
    base = (W + H) / 2.0

    # Buildings: solid rounded squares, fully team-coloured (the classic
    # false-positive for icon detectors). The clean base already has the real
    # static buildings; these extra random-position ones force the detector to
    # reject buildings by APPEARANCE rather than by fixed map location.
    for _ in range(rng.randint(2, 8)):
        team = rng.choice(("ally", "enemy"))
        s = int(base * rng.uniform(0.035, 0.06))
        if s < 3:
            continue
        x = rng.randint(0, max(0, W - s))
        y = rng.randint(0, max(0, H - s))
        col = _sample_unit_color(team, rng)
        cv2.rectangle(canvas, (x, y), (x + s, y + s), col, -1)
        if rng.random() < 0.5:  # slight outline
            cv2.rectangle(canvas, (x, y), (x + s, y + s), (20, 20, 20), 1)

    # Creep waves: tight clusters of tiny dots along a short line.
    for _ in range(rng.randint(1, 4)):
        team = rng.choice(("ally", "enemy"))
        col = _sample_unit_color(team, rng)
        cx, cy = rng.randint(0, W - 1), rng.randint(0, H - 1)
        ang = rng.uniform(0, np.pi)
        dx, dy = np.cos(ang), np.sin(ang)
        for k in range(rng.randint(4, 12)):
            r = max(1, int(base * rng.uniform(0.004, 0.01)))
            px = int(cx + dx * k * base * 0.02 + rng.uniform(-3, 3))
            py = int(cy + dy * k * base * 0.02 + rng.uniform(-3, 3))
            cv2.circle(canvas, (px, py), r, col, -1)

    # Couriers: lone small circles.
    for _ in range(rng.randint(0, 2)):
        team = rng.choice(("ally", "enemy"))
        col = _sample_unit_color(team, rng)
        r = max(2, int(base * rng.uniform(0.012, 0.02)))
        x, y = rng.randint(0, W - 1), rng.randint(0, H - 1)
        cv2.circle(canvas, (x, y), r, col, -1)

    # Neutral camp / ward markers: yellow-ish triangles & dots.
    for _ in range(rng.randint(1, 5)):
        s = max(2, int(base * rng.uniform(0.012, 0.025)))
        x, y = rng.randint(0, W - 1), rng.randint(0, H - 1)
        pts = np.array([[x, y + s], [x + s, y + s], [x + s // 2, y]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, pts, _YELLOW_BGR)


# ---------------------------------------------------------------------------
# Background handling
# ---------------------------------------------------------------------------


def load_backgrounds(paths: list[Path]) -> list[np.ndarray]:
    """Load background crops (BGR). Silently skips unreadable files."""
    bgs: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            bgs.append(img)
    return bgs


def _prep_background(bg: np.ndarray, rng: random.Random, swap_base_p: float = 0.0) -> np.ndarray:
    """Resize to a randomized minimap-like canvas and apply light augmentation.

    ``swap_base_p`` defaults to 0: we rely on the two supplied real captures
    (green-BL and green-TR) for base-corner variety and never rotate the
    terrain (rotation would look unnatural).
    """
    if swap_base_p > 0 and rng.random() < swap_base_p:
        bg = cv2.rotate(bg, cv2.ROTATE_180)
    target = rng.randint(320, 420)
    aspect = rng.uniform(0.92, 1.08)
    W = target
    H = int(round(target * aspect))
    out = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)

    # HSV jitter
    if rng.random() < 0.8:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.randint(-6, 6)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.8, 1.15), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.25:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


# ---------------------------------------------------------------------------
# Frame synthesis
# ---------------------------------------------------------------------------


@dataclass
class SynthConfig:
    team_size: int = 5             # heroes per team (5v5)
    ally_visible_p: float = 0.9    # P(an ally is on the minimap): hidden only when dead/respawning
    enemy_visible_p: float = 0.5   # P(an enemy is visible): fog of war hides most of them
    min_visible: int = 1           # force at least this many icons per frame
    icon_frac_lo: float = 0.072    # icon side as fraction of canvas width (matches real ~28-32px captures)
    icon_frac_hi: float = 0.10
    allow_overlap: bool = True
    max_overlap_iou: float = 0.35  # reject placements above this IoU vs. existing
    margin_frac: float = 0.02
    draw_distractors: bool = False  # real backgrounds already contain real buildings/creeps/camps
    swap_base_p: float = 0.0       # keep 0: use the two real bases as-is, never rotate terrain


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def synth_frame(
    bank: PortraitBank,
    backgrounds: list[np.ndarray],
    shorts: list[str],
    rng: random.Random,
    cfg: SynthConfig | None = None,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Generate one synthetic frame.

    Returns ``(image_bgr, labels)`` where each label is
    ``(class_id, cx, cy, w, h)`` in YOLO-normalized coordinates (0..1).
    """
    cfg = cfg or SynthConfig()
    if not backgrounds:
        raise ValueError("no backgrounds supplied")

    canvas = _prep_background(rng.choice(backgrounds), rng, cfg.swap_base_p).copy()
    H, W = canvas.shape[:2]

    if cfg.draw_distractors:
        _draw_distractors(canvas, rng)

    available = bank.available()
    short_index = {s: i for i, s in enumerate(shorts)}
    margin = int(min(W, H) * cfg.margin_frac)

    # A real match = 10 DISTINCT heroes, 5 per team. Sample the roster, then
    # decide visibility per hero (allies hidden when dead/respawning; enemies
    # hidden by fog of war -> far fewer visible). This yields ~1..10 icons.
    k = min(2 * cfg.team_size, len(available))
    roster = rng.sample(available, k)
    ally_pool = roster[: cfg.team_size]
    enemy_pool = roster[cfg.team_size : 2 * cfg.team_size]
    picks: list[tuple[str, str]] = []
    for s in ally_pool:
        if rng.random() < cfg.ally_visible_p:
            picks.append((s, "ally"))
    for s in enemy_pool:
        if rng.random() < cfg.enemy_visible_p:
            picks.append((s, "enemy"))
    # guarantee a minimum number of visible icons
    if len(picks) < cfg.min_visible:
        hidden = [(s, "ally") for s in ally_pool if (s, "ally") not in picks]
        hidden += [(s, "enemy") for s in enemy_pool if (s, "enemy") not in picks]
        rng.shuffle(hidden)
        picks += hidden[: cfg.min_visible - len(picks)]
    rng.shuffle(picks)

    labels: list[tuple[int, float, float, float, float]] = []
    placed_boxes: list[tuple[float, float, float, float]] = []

    for short, team in picks:
        size = int(W * rng.uniform(cfg.icon_frac_lo, cfg.icon_frac_hi))
        size = max(14, size)

        # try a few positions to respect overlap constraint
        placed = False
        for _try in range(12):
            x = rng.randint(margin, max(margin, W - size - margin))
            y = rng.randint(margin, max(margin, H - size - margin))
            box = (float(x), float(y), float(size), float(size))
            if not cfg.allow_overlap and placed_boxes:
                if any(_iou(box, b) > 0 for b in placed_boxes):
                    continue
            if placed_boxes and max((_iou(box, b) for b in placed_boxes), default=0.0) > cfg.max_overlap_iou:
                continue
            placed = True
            break
        if not placed:
            continue

        portrait_bgr, portrait_alpha = bank.get(short)
        icon_bgr, icon_alpha, pad = render_hero_icon(portrait_bgr, portrait_alpha, size, team, rng)
        # paste so the portrait SQUARE lands at (x, y); the arrow may extend
        # into the surrounding pad on any side.
        _paste_rgba(canvas, icon_bgr, icon_alpha, x - pad, y - pad)

        # label box = the square only (stable target; arrow excluded so the
        # box centre stays exactly on the hero's true minimap position)
        cid = class_id(short_index[short], team)
        cx = (x + size / 2) / W
        cy = (y + size / 2) / H
        labels.append((cid, cx, cy, size / W, size / H))
        placed_boxes.append((float(x), float(y), float(size), float(size)))

    # whole-frame light augmentation
    if rng.random() < 0.2:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return canvas, labels
