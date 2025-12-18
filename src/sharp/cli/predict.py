"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import camera, io
from sharp.utils import net as net_utils
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def _iter_trajectory_variants(
    base_params: camera.TrajectoryParams,
    *,
    enabled: bool,
    count: int,
):
    """Yield (params, output_suffix) for trajectory variants."""
    # Keep this logic local to avoid forcing a shared API surface between CLIs.
    if not enabled:
        yield base_params, ""
        return

    if count < 1:
        raise ValueError("count must be >= 1")

    import dataclasses
    import random

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


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--fps",
    type=float,
    default=60.0,
    show_default=True,
    help="Frames per second for trajectory videos (only used with --render).",
)
@click.option(
    "--duration-scale",
    type=float,
    default=8.0,
    show_default=True,
    help="Scale trajectory length by multiplying the number of frames (only used with --render).",
)
@click.option(
    "--trajectory-variants/--no-trajectory-variants",
    default=False,
    show_default=True,
    help="Render 5 variations of the trajectory (writes 5 output videos; only used with --render).",
)
@click.option(
    "--trajectory-variants-count",
    type=click.IntRange(1, 5),
    default=5,
    show_default=True,
    help="Number of trajectory variations to render (only used with --render).",
)
@click.option(
    "--output-prefix",
    type=str,
    default="",
    help="Prefix for output filenames (e.g. 'previz_'; only used with --render).",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Show per-frame progress (elapsed and ETA; only used with --render).",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    fps: float,
    duration_scale: float,
    trajectory_variants: bool,
    trajectory_variants_count: int,
    output_prefix: str,
    progress: bool,
    device: str,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
        with_rendering = False
    if with_rendering and device == "cuda":
        from sharp.utils import gsplat as gsplat_utils

        if not gsplat_utils.is_gsplat_cuda_extension_available():
            LOGGER.warning(
                "gsplat CUDA extension is not available; rendering is disabled. "
                "Install a CUDA toolkit (nvcc) so gsplat can compile its extension "
                "(on Windows this also requires Visual Studio C++ Build Tools)."
            )
            with_rendering = False

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        net_utils.install_system_certificates()
        try:
            state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download the default checkpoint. If you're behind a corporate proxy/SSL "
                "interceptor, ensure the proxy root certificate is installed in the Windows trust "
                "store, or download the checkpoint manually and pass it via `-c/--checkpoint-path`."
            ) from exc
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )
        gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))

        LOGGER.info("Saving 3DGS to %s", output_path)
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        if with_rendering:
            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
            base_params = camera.TrajectoryParams()
            variants_enabled = trajectory_variants and trajectory_variants_count > 1
            for variant_index, (params, suffix) in enumerate(
                _iter_trajectory_variants(
                    base_params, enabled=variants_enabled, count=trajectory_variants_count
                )
            ):
                output_video_path = output_path / f"{output_prefix}{image_path.stem}{suffix}.mp4"
                if variants_enabled:
                    LOGGER.info("Rendering trajectory variant %d to %s", variant_index, output_video_path)
                else:
                    LOGGER.info("Rendering trajectory to %s", output_video_path)
                render_gaussians(
                    gaussians,
                    metadata,
                    output_video_path,
                    params=params,
                    fps=fps,
                    duration_scale=duration_scale,
                    progress=progress,
                )


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians
