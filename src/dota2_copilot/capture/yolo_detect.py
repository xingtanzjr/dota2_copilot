"""YOLO minimap hero detector (``display_mode == "yolo"``).

Runs a trained Ultralytics YOLOv8 model (``models/minimap_yolo.pt``) over the
minimap crop. The model predicts 254 classes = 127 heroes x {ally, enemy}
(see :mod:`dota2_copilot.training.classes`), so every detection decodes
directly to a hero ``short`` id AND a player-perspective team. No separate
template matching or HSV team-ring sampling is needed.

``ultralytics`` / ``torch`` are imported lazily inside :meth:`_ensure_model`
so the base app (and the minimal py3.8 tooling env) can import this module
without those heavy dependencies installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, YoloDetectConfig
from ..training.classes import decode_class, load_hero_shorts
from ..types import HeroBlob, Point, Team


class YoloMinimapDetector:
    """Lazy-loaded YOLOv8 detector returning ``(enemies, allies)`` HeroBlobs."""

    def __init__(self, cfg: YoloDetectConfig, roster: set[str] | None = None) -> None:
        self.cfg = cfg
        self._model = None
        self._shorts = load_hero_shorts()
        # Optional restriction to the picked heroes (short ids). None = keep all.
        self._roster: set[str] | None = set(roster) if roster else None

    # ------------------------------------------------------------------
    # Roster injection
    # ------------------------------------------------------------------

    def set_roster(self, roster: set[str] | None) -> None:
        """Restrict kept detections to ``roster`` heroes (``None`` = keep all)."""
        self._roster = set(roster) if roster else None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy: heavy import (torch)

            weights = Path(self.cfg.weights)
            if not weights.is_absolute():
                weights = REPO_ROOT / weights
            if not weights.exists():
                raise FileNotFoundError(
                    f"YOLO weights not found: {weights}. Train first with "
                    "scripts/train_yolo.py (see docs/design.md) or point "
                    "config minimap.yolo.weights at an existing .pt file."
                )
            self._model = YOLO(str(weights))
        return self._model

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def detect(self, minimap_bgr: np.ndarray) -> tuple[list[HeroBlob], list[HeroBlob]]:
        if minimap_bgr.ndim != 3 or minimap_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR image, got shape {minimap_bgr.shape}")

        model = self._ensure_model()
        h, w = minimap_bgr.shape[:2]

        results = model.predict(
            source=minimap_bgr,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device or None,
            max_det=self.cfg.max_det,
            half=self.cfg.half,
            verbose=False,
        )

        enemies: list[HeroBlob] = []
        allies: list[HeroBlob] = []
        if not results:
            return enemies, allies

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return enemies, allies

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), cid, score in zip(xyxy, cls, conf):
            short, team_str = decode_class(int(cid), self._shorts)
            if self._roster is not None and short not in self._roster:
                continue
            team = Team.ALLY if team_str == "ally" else Team.ENEMY
            bx, by = int(round(x1)), int(round(y1))
            bw, bh = int(round(x2 - x1)), int(round(y2 - y1))
            cx = int(round((x1 + x2) / 2))
            cy = int(round((y1 + y2) / 2))
            blob = HeroBlob(
                team=team,
                pos=Point(x=cx / w, y=cy / h),
                pixel_pos=(cx, cy),
                bbox=(bx, by, bw, bh),
                area=bw * bh,
                hero_id=short,
                score=float(score),
            )
            (allies if team == Team.ALLY else enemies).append(blob)

        return enemies, allies
