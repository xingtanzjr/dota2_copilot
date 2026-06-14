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
    from_image: Optional[Path] = typer.Option(
        None,
        "--from-image",
        "-i",
        help=(
            "Calibrate from an image file instead of live capture. "
            "Take a screenshot with Win+Shift+S, save it, then pass its path."
        ),
    ),
    delay: int = typer.Option(
        3,
        "--delay",
        "-d",
        help="Seconds to wait before grabbing the screen (so you can Alt+Tab back to Dota).",
    ),
) -> None:
    """Select the minimap screen region and save calibration."""
    from .tools.calibrate_minimap import run_calibration

    run_calibration(out_path=out, from_image=from_image, delay=delay)


@app.command("calibrate-topbar")
def calibrate_topbar(
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Override output path for topbar.json."
    ),
    from_image: Optional[Path] = typer.Option(
        None, "--from-image", "-i",
        help="Calibrate from a screenshot file instead of live capture.",
    ),
    delay: int = typer.Option(
        3, "--delay", "-d",
        help="Seconds to wait before grabbing the screen (so you can Alt+Tab back to Dota).",
    ),
) -> None:
    """Mark the 10-hero strip at the top of the HUD (improves roster accuracy).

    Drag a rectangle that encloses all 10 hero portraits (Radiant #1 through
    Dire #5), including the clock gap. Vertically, include only the portrait
    area -- skip the HP/gold bars below. Result -> config/topbar.json.
    """
    from .tools.calibrate_topbar import run_topbar_calibration

    run_topbar_calibration(out_path=out, from_image=from_image, delay=delay)


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
    fps: Optional[float] = typer.Option(
        5.0,
        "--fps",
        help="Override capture FPS for preview. Higher = smoother but more CPU.",
    ),
    detect_roster: bool = typer.Option(
        True,
        "--detect-roster/--no-detect-roster",
        help="Auto-detect the 10 heroes from the top bar at startup (10-15x speedup).",
    ),
    my_team: Optional[str] = typer.Option(
        None, "--my-team", "-t",
        help="Which side you're on: 'radiant' or 'dire'. Lets the analyzer skip per-frame team detection.",
    ),
    roster_delay: int = typer.Option(
        5, "--roster-delay",
        help="Seconds to wait before grabbing the top bar for roster detection.",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to app.yaml (defaults to repo config/app.yaml)."
    ),
) -> None:
    """Live preview: continuously detect heroes on the minimap.

    At startup, by default, the top bar is captured and the 10 heroes are
    identified. This restricts subsequent template matching to those heroes
    only -- typically 10-15x faster than matching all 127. Pass --no-detect-roster
    to skip (useful during a replay or if the top bar is obscured).
    """
    from .config import load_app_config
    from .tools.debug_preview import run_preview

    if my_team is not None and my_team not in ("radiant", "dire"):
        raise typer.BadParameter("--my-team must be 'radiant' or 'dire'")
    cfg = load_app_config(config) if config else None
    run_preview(
        config=cfg,
        scale=scale,
        fps_override=fps,
        detect_roster=detect_roster,
        my_team=my_team,
        roster_delay=roster_delay,
    )


@app.command()
def dump(
    delay: int = typer.Option(
        3, "--delay", "-d",
        help="Seconds to wait before grabbing (Alt+Tab back to Dota in the meantime).",
    ),
    from_image: Optional[Path] = typer.Option(
        None, "--from-image", "-i",
        help="Read from a full-resolution screenshot instead of live capture.",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to app.yaml."
    ),
) -> None:
    """Grab ONE frame, run detection, write a full debug bundle, then exit."""
    from .config import load_app_config
    from .tools.debug_preview import run_dump

    cfg = load_app_config(config) if config else None
    run_dump(config=cfg, delay=delay, from_image=from_image)


@app.command()
def roster(
    delay: int = typer.Option(
        5, "--delay", "-d",
        help="Seconds to wait before grabbing the screen.",
    ),
    from_image: Optional[Path] = typer.Option(
        None, "--from-image", "-i",
        help="Read from a full screenshot instead of live capture.",
    ),
    my_team: Optional[str] = typer.Option(
        None, "--my-team", "-t",
        help="Which side you're on: 'radiant' or 'dire'. Stored in roster.json.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Override output path for roster.json."
    ),
) -> None:
    """Detect the 10 heroes from the top bar and save them to config/roster.json.

    Once a roster is saved, the minimap analyzer restricts template matching
    to those 10 heroes only -- a 10-15x speedup vs. matching all 127.
    Run this once at the start of each match (after all heroes are picked).
    """
    from .tools.detect_roster import run_roster_detection

    if my_team is not None and my_team not in ("radiant", "dire"):
        raise typer.BadParameter("--my-team must be 'radiant' or 'dire'")
    run_roster_detection(delay=delay, from_image=from_image, out_path=out, my_team=my_team)


@app.command("learn-topbar")
def learn_topbar(
    assignments: list[str] = typer.Argument(
        ...,
        help=(
            "One or more 'slot=hero' pairs, e.g. R3=pudge or R3=小小. "
            "Slots: R1..R5 (Radiant), D1..D5 (Dire). "
            "Hero accepts Chinese name, English name, or short id."
        ),
    ),
    from_image: Optional[Path] = typer.Option(
        None, "--from-image", "-i",
        help="Read from a full screenshot instead of live capture (recommended).",
    ),
    delay: int = typer.Option(
        5, "--delay", "-d",
        help="Countdown seconds before live screen capture (if --from-image is omitted).",
    ),
    label: str = typer.Option(
        "variant", "--label", "-l",
        help="Variant tag stored in the filename, e.g. 'arcana', 'persona_toy_butcher'.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing variant files with the same name.",
    ),
) -> None:
    """Teach the detector new topbar portraits (Arcana / Persona / Immortal alts).

    Requires a saved top-bar calibration (run `calibrate-topbar` first). Each
    side's calibrated rect is split into 5 equal slots; for every slot you
    label, the tool crops that region and saves it to
    `assets/topbar_variants/<hero>__<label>.png`. The roster detector
    automatically uses these variants alongside the canonical portraits.

    Hero names accept Chinese (敌法师 / 幻影刺客), English (Anti-Mage),
    or the canonical short id (antimage).

    Examples
    --------
        # Pudge with the Feast of Abscession arcana at Radiant #3,
        # PA with Manifold Paradox at Dire #1:
        dota2-copilot learn-topbar R3=pudge D1=phantom_assassin \\
            --from-image game.png --label arcana

        # 用中文也可以：
        dota2-copilot learn-topbar R3=小小 D1=幻影刺客 \\
            --from-image game.png --label arcana
    """
    from .tools.learn_topbar import run_learn_topbar

    n = run_learn_topbar(
        assignments=assignments,
        from_image=from_image,
        delay=delay,
        label=label,
        force=force,
    )
    if n == 0:
        raise typer.Exit(code=1)


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
