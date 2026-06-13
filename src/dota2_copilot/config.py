"""Configuration loading.

Two layers of configuration:
1. `app.yaml`     — versioned defaults checked into the repo (config/app.yaml).
2. `minimap.json` — produced by the calibration tool, contains user-specific
                    screen coordinates of the minimap region. Not versioned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .types import ScreenRect

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APP_CONFIG = REPO_ROOT / "config" / "app.yaml"
DEFAULT_MINIMAP_CONFIG = REPO_ROOT / "config" / "minimap.json"
DEFAULT_TOPBAR_CONFIG = REPO_ROOT / "config" / "topbar.json"
DEFAULT_LANDMARKS_TEMPLATE = REPO_ROOT / "assets" / "map_landmarks.yaml"
DEFAULT_LANDMARKS_OVERRIDE = REPO_ROOT / "config" / "map_landmarks.json"
DEFAULT_ROSTER_PATH = REPO_ROOT / "config" / "roster.json"
HEROES_JSON = REPO_ROOT / "assets" / "heroes.json"


# ---------------------------------------------------------------------------
# Hero name lookup (short -> Chinese / English display names)
# ---------------------------------------------------------------------------

_HERO_NAMES_CACHE: dict[str, dict[str, str]] | None = None


def _load_hero_names() -> dict[str, dict[str, str]]:
    """Return {short: {"zh": ..., "en": ...}} from assets/heroes.json."""
    global _HERO_NAMES_CACHE
    if _HERO_NAMES_CACHE is not None:
        return _HERO_NAMES_CACHE
    out: dict[str, dict[str, str]] = {}
    if HEROES_JSON.exists():
        data = json.loads(HEROES_JSON.read_text(encoding="utf-8"))
        for h in data.get("heroes", []):
            short = h.get("short")
            if not short:
                continue
            out[short] = {
                "zh": h.get("name_zh") or h.get("localized_name") or short,
                "en": h.get("localized_name") or short,
            }
    _HERO_NAMES_CACHE = out
    return out


def hero_zh(short: str) -> str:
    """Chinese display name for a hero short id (falls back to short)."""
    return _load_hero_names().get(short, {}).get("zh", short)


def hero_label(short: str) -> str:
    """`中文名 (short)` — useful when both are wanted in a single line."""
    zh = hero_zh(short)
    return f"{zh} ({short})" if zh != short else short


# ---------------------------------------------------------------------------
# Pydantic models mirroring config/app.yaml
# ---------------------------------------------------------------------------


class RecordConfig(BaseModel):
    enabled: bool = False
    out_dir: str = "recordings"


class CaptureConfig(BaseModel):
    fps: float = 1.0
    history_seconds: float = 30.0
    record: RecordConfig = Field(default_factory=RecordConfig)


class DecisionConfig(BaseModel):
    backend: str = "rule"


class ChannelConfig(BaseModel):
    enabled: bool = False
    min_level: str = "info"


class NotifierConfig(BaseModel):
    channels: dict[str, ChannelConfig] = Field(default_factory=dict)


class ColorRangeConfig(BaseModel):
    h_ranges: list[tuple[int, int]]
    s_min: int = 0
    v_min: int = 0


class IconsDetectConfig(BaseModel):
    fill_kernel: int = 3
    icon_area_min: int = 500
    icon_area_max: int = 2500
    aspect_tol: float = 1.6
    min_fill_ratio: float = 0.45
    building_raw_fill_max: float = 0.55


class TemplateDetectConfig(BaseModel):
    """Parameters for template-matching hero detection.

    Templates are loaded from ``assets/minimap/<short>_<size>.png``.
    """

    template_dir: str = "assets/minimap"
    template_size: int = 32                 # which size suffix to use (28/30/32)
    score_threshold: float = 0.55           # min matchTemplate response to keep
    nms_distance: int = 16                  # px between centers for same-hero NMS
    team_ring_thickness: int = 7            # px ring outside bbox for team color
    team_red_h_ranges: list[tuple[int, int]] = Field(default_factory=lambda: [(0, 12), (165, 180)])
    team_green_h_ranges: list[tuple[int, int]] = Field(default_factory=lambda: [(38, 90)])
    team_s_min: int = 70
    team_v_min: int = 70
    team_min_pixels: int = 6                # below this in BOTH colors -> unknown
    # NOTE: roster is set at RUNTIME by the roster-detection step (see
    # tools/detect_roster.py), not from config. It's an in-memory optimization.


class MinimapDetectConfig(BaseModel):
    display_mode: Literal["icons", "icons_template", "names", "arrows"] = "icons_template"
    enemy_red: ColorRangeConfig
    ally_green: ColorRangeConfig
    icons: IconsDetectConfig = Field(default_factory=IconsDetectConfig)
    template: TemplateDetectConfig = Field(default_factory=TemplateDetectConfig)


class AppConfig(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    notifier: NotifierConfig = Field(default_factory=NotifierConfig)
    minimap: MinimapDetectConfig


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load and validate the application configuration."""

    path = path or DEFAULT_APP_CONFIG
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


def load_roster(path: Path | None = None, silent: bool = False) -> list[str] | None:
    """Optional helper: read a previously saved roster.json (debug use only).

    The normal runtime flow is to detect the roster at startup and keep it in
    memory; this helper is only useful for offline experiments.
    """
    path = path or DEFAULT_ROSTER_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        heroes = data.get("heroes") or []
        return [str(h) for h in heroes] if heroes else None
    except (json.JSONDecodeError, OSError) as e:
        if not silent:
            print(f"[config] failed to read {path}: {e}")
        return None


def save_roster(heroes: list[str], path: Path | None = None) -> Path:
    """Optional helper: persist a roster for debugging (not auto-loaded)."""
    path = path or DEFAULT_ROSTER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"heroes": list(heroes)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Minimap calibration (separate file, written by the calibrate tool)
# ---------------------------------------------------------------------------


class _ScreenRectModel(BaseModel):
    x: int
    y: int
    width: int
    height: int


class MinimapCalibration(BaseModel):
    """Screen coordinates of the minimap region produced by calibrate_minimap."""

    screen_width: int
    screen_height: int
    minimap: _ScreenRectModel

    def rect(self) -> ScreenRect:
        return ScreenRect(
            x=self.minimap.x,
            y=self.minimap.y,
            width=self.minimap.width,
            height=self.minimap.height,
        )

    @classmethod
    def from_rect(
        cls, screen_width: int, screen_height: int, rect: ScreenRect
    ) -> MinimapCalibration:
        return cls(
            screen_width=screen_width,
            screen_height=screen_height,
            minimap=_ScreenRectModel(
                x=rect.x, y=rect.y, width=rect.width, height=rect.height
            ),
        )


def load_minimap_calibration(path: Path | None = None) -> MinimapCalibration:
    path = path or DEFAULT_MINIMAP_CONFIG
    if not path.exists():
        raise FileNotFoundError(
            f"Minimap calibration not found at {path}. "
            "Run `dota2-copilot calibrate` first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return MinimapCalibration.model_validate(raw)


def save_minimap_calibration(cal: MinimapCalibration, path: Path | None = None) -> Path:
    path = path or DEFAULT_MINIMAP_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cal.model_dump_json(indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Top-bar calibration (location of the 10-hero strip at the top of the HUD)
# ---------------------------------------------------------------------------


class TopbarCalibration(BaseModel):
    """Screen coordinates of the top-bar hero strips.

    Two separate rectangles -- one for the Radiant 5 portraits on the left and
    one for the Dire 5 portraits on the right -- so the clock / score area in
    the middle is excluded entirely. Vertically include only the portrait
    area (NOT the HP/gold bars below).
    """

    screen_width: int
    screen_height: int
    radiant: _ScreenRectModel
    dire: _ScreenRectModel

    def radiant_rect(self) -> ScreenRect:
        return ScreenRect(
            x=self.radiant.x, y=self.radiant.y,
            width=self.radiant.width, height=self.radiant.height,
        )

    def dire_rect(self) -> ScreenRect:
        return ScreenRect(
            x=self.dire.x, y=self.dire.y,
            width=self.dire.width, height=self.dire.height,
        )

    @classmethod
    def from_rects(
        cls,
        screen_width: int,
        screen_height: int,
        radiant: ScreenRect,
        dire: ScreenRect,
    ) -> "TopbarCalibration":
        return cls(
            screen_width=screen_width,
            screen_height=screen_height,
            radiant=_ScreenRectModel(
                x=radiant.x, y=radiant.y, width=radiant.width, height=radiant.height
            ),
            dire=_ScreenRectModel(
                x=dire.x, y=dire.y, width=dire.width, height=dire.height
            ),
        )


def load_topbar_calibration(path: Path | None = None) -> TopbarCalibration | None:
    """Return the saved top-bar calibration, or None if not yet calibrated."""
    path = path or DEFAULT_TOPBAR_CONFIG
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TopbarCalibration.model_validate(raw)


def save_topbar_calibration(cal: TopbarCalibration, path: Path | None = None) -> Path:
    path = path or DEFAULT_TOPBAR_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cal.model_dump_json(indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Map landmarks (fixed minimap features — towers, runes, Roshan, etc.)
# ---------------------------------------------------------------------------


class LandmarkDefaults(BaseModel):
    mask_radius: float = 0.04


class MapLandmarks(BaseModel):
    """Normalized minimap coordinates for fixed map features.

    Loaded from ``assets/map_landmarks.yaml`` (committed defaults); per-user
    calibration overrides are merged from ``config/map_landmarks.json`` when
    present.
    """

    version: int = 1
    patch: str = "7.x"
    defaults: LandmarkDefaults = Field(default_factory=LandmarkDefaults)
    landmarks: dict[str, tuple[float, float]] = Field(default_factory=dict)

    def names(self) -> list[str]:
        return list(self.landmarks.keys())

    def to_pixels(
        self, minimap_w: int, minimap_h: int
    ) -> dict[str, tuple[int, int]]:
        """Convert all normalized coords to pixel coords for a given minimap size."""
        return {
            name: (int(round(x * minimap_w)), int(round(y * minimap_h)))
            for name, (x, y) in self.landmarks.items()
        }


def load_map_landmarks(
    template_path: Path | None = None,
    override_path: Path | None = None,
) -> MapLandmarks:
    """Load landmarks from the YAML template, then merge JSON overrides if any."""
    template_path = template_path or DEFAULT_LANDMARKS_TEMPLATE
    override_path = override_path or DEFAULT_LANDMARKS_OVERRIDE

    with template_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    model = MapLandmarks.model_validate(base)

    if override_path.exists():
        raw = json.loads(override_path.read_text(encoding="utf-8"))
        # Merge: override values replace template values key-by-key; missing
        # keys in the override fall back to the template.
        merged_landmarks = dict(model.landmarks)
        for k, v in (raw.get("landmarks") or {}).items():
            merged_landmarks[k] = (float(v[0]), float(v[1]))
        model = model.model_copy(update={"landmarks": merged_landmarks})

    return model


def save_map_landmarks_override(
    landmarks: dict[str, tuple[float, float]],
    path: Path | None = None,
) -> Path:
    """Persist user-calibrated landmark coords to config/map_landmarks.json."""
    path = path or DEFAULT_LANDMARKS_OVERRIDE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "landmarks": {k: [float(x), float(y)] for k, (x, y) in landmarks.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
