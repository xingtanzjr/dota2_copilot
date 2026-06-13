"""CLI entry point.

Subcommands available in Milestone 1:
    calibrate            — interactively pick the minimap region (one-time setup)
    calibrate-landmarks  — click-to-locate fixed map features (towers, runes, …)
    preview              — live OpenCV window with detection overlay (debugging)
    record               — capture frames + detections to disk for offline replay
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    add_completion=False,
    help="Dota 2 Copilot — minimap-aware reminder assistant.",
    no_args_is_help=True,
)


@app.command()
def calibrate(
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Override output path for minimap.json."
    ),
) -> None:
    """Select the minimap screen region and save calibration."""
    from .tools.calibrate_minimap import run_calibration

    run_calibration(out_path=out)


@app.command("calibrate-landmarks")
def calibrate_landmarks(
    from_image: Optional[Path] = typer.Option(
        None,
        "--from-image",
        "-i",
        help=(
            "Calibrate from an image file instead of live screen capture. "
            "Pass a cropped minimap or a full screenshot."
        ),
    ),
    scale: int = typer.Option(
        3, "--scale", help="Display zoom factor for clickability."
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Override output path for map_landmarks.json."
    ),
) -> None:
    """Click-to-locate fixed map features (towers, runes, Roshan, etc.)."""
    from .tools.calibrate_landmarks import run_landmark_calibration

    run_landmark_calibration(from_image=from_image, scale=scale, out_path=out)


@app.command()
def preview(
    scale: float = typer.Option(2.0, "--scale", help="Display zoom factor."),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to app.yaml (defaults to repo config/app.yaml)."
    ),
) -> None:
    """Live preview: continuously detect heroes on the minimap."""
    from .config import load_app_config
    from .tools.debug_preview import run_preview

    cfg = load_app_config(config) if config else None
    run_preview(config=cfg, scale=scale)


@app.command()
def record(
    duration: Optional[float] = typer.Option(
        None, "--duration", "-d", help="Seconds to record. Omit for unbounded (Ctrl-C to stop)."
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Session name. Defaults to UTC timestamp."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to app.yaml."
    ),
) -> None:
    """Record minimap frames + detection results to disk."""
    from .config import load_app_config
    from .tools.record import run_record

    cfg = load_app_config(config) if config else None
    run_record(duration_seconds=duration, session_name=name, config=cfg)


if __name__ == "__main__":
    app()
