"""Training-side tooling for the YOLO minimap detector.

This subpackage is intentionally decoupled from the runtime app: it only needs
``numpy``, ``opencv``, ``pyyaml`` and (for training) ``ultralytics``. It does
*not* import :mod:`dota2_copilot.config` (pydantic) so the synthetic-data
generator can run in a minimal environment.

Modules
-------
* :mod:`.classes` — the canonical 254-class scheme (127 heroes x {ally, enemy})
  and ``data.yaml`` emission for Ultralytics.
* :mod:`.synth`   — synthetic minimap frame + YOLO-label generator.
"""
