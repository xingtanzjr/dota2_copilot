"""Shared data types used across capture, state, decision and notifier layers.

Coordinates convention
----------------------
* `ScreenRect`  : absolute pixel rectangle on the user's monitor (top-left origin).
* `pixel coord` : (x, y) in pixels within a cropped minimap image.
* `norm coord`  : (x, y) ∈ [0, 1] within the minimap. This is the canonical
                  representation passed to state / decision layers so they are
                  resolution-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np


class Team(str, Enum):
    ALLY = "ally"
    ENEMY = "enemy"
    UNKNOWN = "unknown"


class Lane(str, Enum):
    TOP = "top"
    MID = "mid"
    BOT = "bot"


@dataclass(slots=True, frozen=True)
class Point:
    """Normalized (0..1) point on the minimap."""

    x: float
    y: float


@dataclass(slots=True, frozen=True)
class ScreenRect:
    """Absolute pixel rectangle on the monitor."""

    x: int
    y: int
    width: int
    height: int

    def as_mss_dict(self) -> dict[str, int]:
        return {"left": self.x, "top": self.y, "width": self.width, "height": self.height}


@dataclass(slots=True)
class HeroBlob:
    """A detected hero portrait icon on the minimap.

    Detection is based on the team-colored border around each hero portrait,
    so `pos` / `pixel_pos` are the center of the icon's bounding box.
    """

    team: Team
    pos: Point                          # normalized icon center within minimap
    pixel_pos: tuple[int, int]          # icon center, pixel coords within the crop
    bbox: tuple[int, int, int, int]     # (x, y, w, h) in pixel coords within the crop
    area: int                           # contour area (filled), pixels


@dataclass(slots=True)
class Frame:
    """A single sampled minimap frame plus detection results."""

    timestamp: float                     # wall clock seconds (time.time())
    enemies: list[HeroBlob] = field(default_factory=list)
    allies: list[HeroBlob] = field(default_factory=list)
    minimap_size: tuple[int, int] = (0, 0)  # (width, height) of the minimap crop
    raw_image: np.ndarray | None = None     # BGR; kept only when recording


AlertLevel = Literal["info", "warn", "danger", "critical"]
