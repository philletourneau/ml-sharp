"""Contains `sharp render` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
import random
from pathlib import Path

import click
import torch
import torch.utils.data

from sharp.utils import camera, gsplat, io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, load_ply

LOGGER = logging.getLogger(__name__)

_PROGRESS_UPDATE_INTERVAL_S = 2.0


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes:d}m{rem:02.0f}s"
    hours = int(minutes // 60)
    rem_m = minutes - hours * 60
    return f"{hours:d}h{rem_m:02d}m"


def _iter_with_progress(iterable, *, total: int, desc: str, enabled: bool):
    """Iterate with elapsed/ETA progress output."""
    if not enabled:
        yield from iterable
        return

    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]

        yield from tqdm(
            iterable,
            total=total,
            desc=desc,
            unit="frame",
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return
    except Exception:
        pass

    start_t = time.perf_counter()
    last_print_t = start_t
    for index, item in enumerate(iterable, start=1):
        now_t = time.perf_counter()
        should_print = (
            index == 1
            or (now_t - last_print_t) >= _PROGRESS_UPDATE_INTERVAL_S
            or index == total
        )
        if should_print:
            elapsed_s = now_t - start_t
            rate = index / elapsed_s if elapsed_s > 0 else 0.0
            remaining = max(0, total - index)
            eta_s = (remaining / rate) if rate > 0 else float("inf")
            click.echo(
                f"{desc}: {index}/{total} frames | elapsed {_format_duration(elapsed_s)} | "
                f"ETA {_format_duration(eta_s)}",
                err=True,
            )
            last_print_t = now_t
        yield item


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the ply or a list of plys.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the rendered videos.",
    required=True,
)
@click.option(
    "--fps",
    type=float,
    default=60.0,
    show_default=True,
    help="Frames per second for the output videos.",
)
@click.option(
    "--duration-scale",
    type=float,
    default=8.0,
    show_default=True,
    help="Scale trajectory length by multiplying the number of frames.",
)
@click.option(
    "--trajectory-variants/--no-trajectory-variants",
    default=False,
    show_default=True,
    help="Render 5 variations of the trajectory (writes 5 output videos).",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Show per-frame progress (elapsed and ETA).",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def render_cli(
    input_path: Path,
    output_path: Path,
    fps: float,
    duration_scale: float,
    trajectory_variants: bool,
    progress: bool,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if not torch.cuda.is_available():
        LOGGER.error("Rendering a checkpoint requires CUDA.")
        exit(1)

    try:
        gsplat.require_gsplat_cuda_extension()
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        exit(1)

    output_path.mkdir(exist_ok=True, parents=True)

    base_params = camera.TrajectoryParams()

    if input_path.suffix == ".ply":
        scene_paths = [input_path]
    elif input_path.is_dir():
        scene_paths = list(input_path.glob("*.ply"))
    else:
        LOGGER.error("Input path must be either directory or single PLY file.")
        exit(1)

    for scene_path in scene_paths:
        LOGGER.info("Rendering %s", scene_path)
        gaussians, metadata = load_ply(scene_path)
        for variant_index, (params, suffix) in enumerate(
            _iter_trajectory_variants(base_params, enabled=trajectory_variants, count=5)
        ):
            output_video = output_path / f"{scene_path.stem}{suffix}.mp4"
            if trajectory_variants:
                LOGGER.info("Trajectory variant %d -> %s", variant_index, output_video)
            render_gaussians(
                gaussians=gaussians,
                metadata=metadata,
                params=params,
                output_path=output_video,
                fps=fps,
                duration_scale=duration_scale,
                progress=progress,
            )


def render_gaussians(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    params: camera.TrajectoryParams | None = None,
    fps: float = 60.0,
    duration_scale: float = 8.0,
    progress: bool = True,
) -> None:
    """Render a single gaussian checkpoint file."""
    (width, height) = metadata.resolution_px
    f_px = metadata.focal_length_px

    if params is None:
        params = camera.TrajectoryParams()

    if duration_scale <= 0:
        raise ValueError("duration_scale must be > 0.")

    scaled_steps = max(1, int(round(params.num_steps * duration_scale)))
    params = dataclasses.replace(params, num_steps=scaled_steps)

    if not torch.cuda.is_available():
        raise RuntimeError("Rendering a checkpoint requires CUDA.")

    gsplat.require_gsplat_cuda_extension()

    device = torch.device("cuda")

    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2., 0],
            [0, f_px, (height - 1) / 2., 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    camera_model = camera.create_camera_model(
        gaussians, intrinsics, resolution_px=metadata.resolution_px
    )

    trajectory = camera.create_eye_trajectory(
        gaussians, params, resolution_px=metadata.resolution_px, f_px=f_px
    )
    total_frames = len(trajectory)
    LOGGER.info(
        "Trajectory: %d frames @ %.1f fps (~%s)",
        total_frames,
        fps,
        _format_duration(total_frames / fps if fps > 0 else 0.0),
    )
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
    video_writer = io.VideoWriter(output_path, fps=fps)

    for eye_position in _iter_with_progress(
        trajectory, total=total_frames, desc=str(output_path.name), enabled=progress
    ):
        camera_info = camera_model.compute(eye_position)
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=camera_info.extrinsics[None].to(device),
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        color = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        depth = rendering_output.depth[0]
        video_writer.add_frame(color, depth)
    video_writer.close()


def _iter_trajectory_variants(
    base_params: camera.TrajectoryParams,
    *,
    enabled: bool,
    count: int,
):
    """Yield (params, output_suffix) for trajectory variants."""
    if not enabled:
        yield base_params, ""
        return

    if count < 1:
        raise ValueError("count must be >= 1")

    # Use a fixed set of different motion types to ensure variants are visibly distinct.
    variant_types: list[camera.TrajetoryType] = [
        "rotate_forward",
        "rotate",
        "swipe",
        "shake",
        "rotate_forward",
    ]

    for variant_index in range(count):
        rng = random.Random(variant_index)
        disparity_scale = rng.uniform(0.6, 1.6)
        zoom_scale = rng.uniform(0.6, 1.6)
        distance_offset = rng.uniform(-0.2, 0.2)
        repeats = rng.choice([1, 2, 3])
        lookat_mode: camera.LookAtMode = rng.choice(["point", "ahead"])
        traj_type = variant_types[variant_index % len(variant_types)]
        params = dataclasses.replace(
            base_params,
            type=traj_type,
            lookat_mode=lookat_mode,
            max_disparity=base_params.max_disparity * disparity_scale,
            max_zoom=base_params.max_zoom * zoom_scale,
            distance_m=base_params.distance_m + distance_offset,
            num_repeats=repeats,
        )
        yield params, f"_v{variant_index:02d}"
