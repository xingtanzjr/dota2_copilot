"""Thin wrapper around `mss` for fast region screen-capture.

`mss` is cross-platform (Windows / Linux / macOS) and returns BGRA bytes in
~1ms per frame. We convert to BGR (OpenCV's native order) and drop alpha.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import mss
import numpy as np

from ..types import ScreenRect


class ScreenGrabber:
    """Reusable screen grabber. Create once, call `grab()` per frame."""

    def __init__(self) -> None:
        # mss objects are not thread-safe; each thread should own one.
        self._sct = mss.mss()

    # ------------------------------------------------------------------
    # Primary capture API
    # ------------------------------------------------------------------

    def grab(self, rect: ScreenRect) -> np.ndarray:
        """Capture a sub-rectangle of the primary monitor.

        Returns a BGR uint8 array of shape (H, W, 3).
        """
        raw = self._sct.grab(rect.as_mss_dict())
        # raw.raw is BGRA bytes; reshape and drop alpha.
        arr = np.frombuffer(raw.raw, dtype=np.uint8).reshape(raw.height, raw.width, 4)
        return arr[:, :, :3].copy()  # copy to detach from mss buffer

    def grab_full(self, monitor_index: int = 1) -> np.ndarray:
        """Capture an entire monitor (1 = primary in mss convention)."""
        mon = self._sct.monitors[monitor_index]
        raw = self._sct.grab(mon)
        arr = np.frombuffer(raw.raw, dtype=np.uint8).reshape(raw.height, raw.width, 4)
        return arr[:, :, :3].copy()

    def primary_size(self) -> tuple[int, int]:
        """Return (width, height) of the primary monitor."""
        mon = self._sct.monitors[1]
        return mon["width"], mon["height"]

    def close(self) -> None:
        self._sct.close()


@contextmanager
def open_grabber() -> Iterator[ScreenGrabber]:
    g = ScreenGrabber()
    try:
        yield g
    finally:
        g.close()
