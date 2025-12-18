"""Contains `sharp mesh` CLI implementation."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import click
import torch

from sharp.utils import camera, gsplat
from sharp.utils import logging as logging_utils
from sharp.utils.meshing import MeshFromGaussiansOptions, gaussians_ply_to_mesh_tsdf

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a SHARP Gaussians .ply file.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to write the output mesh (.ply recommended).",
    required=True,
)
@click.option(
    "--max-render-dim",
    type=int,
    default=640,
    show_default=True,
    help="Max width/height used when rendering RGBD views for TSDF fusion.",
)
@click.option(
    "--depth-trunc",
    type=float,
    default=10.0,
    show_default=True,
    help="Depth truncation (meters) for TSDF integration.",
)
@click.option(
    "--voxel-length",
    type=float,
    default=0.01,
    show_default=True,
    help="Voxel size (meters) for TSDF integration.",
)
@click.option(
    "--sdf-trunc",
    type=float,
    default=0.04,
    show_default=True,
    help="SDF truncation (meters) for TSDF integration.",
)
@click.option(
    "--trajectory-type",
    type=click.Choice(["swipe", "shake", "rotate", "rotate_forward"], case_sensitive=False),
    default="rotate",
    show_default=True,
    help="Camera trajectory type used to generate viewpoints.",
)
@click.option(
    "--lookat-mode",
    type=click.Choice(["point", "ahead"], case_sensitive=False),
    default="point",
    show_default=True,
    help="Look-at mode override.",
)
@click.option("--max-disparity", type=float, default=None, help="Override trajectory max disparity.")
@click.option("--max-zoom", type=float, default=None, help="Override trajectory max zoom.")
@click.option("--distance-m", type=float, default=None, help="Override trajectory camera distance in meters.")
@click.option(
    "--num-steps",
    type=int,
    default=240,
    show_default=True,
    help="Trajectory number of steps (views per repeat).",
)
@click.option(
    "--num-repeats",
    type=int,
    default=1,
    show_default=True,
    help="Trajectory number of repeats.",
)
@click.option(
    "--low-pass-filter-eps",
    type=float,
    default=0.0,
    show_default=True,
    help="Low-pass filter epsilon for gsplat rasterization.",
)
@click.option(
    "--cuda-device",
    type=int,
    default=0,
    show_default=True,
    help="CUDA device index to use.",
)
@click.option(
    "--target-triangles",
    type=int,
    default=None,
    help="If set, simplify the extracted mesh to approximately this many triangles.",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Print periodic progress while integrating views.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def mesh_cli(
    input_path: Path,
    output_path: Path,
    max_render_dim: int,
    depth_trunc: float,
    voxel_length: float,
    sdf_trunc: float,
    trajectory_type: str,
    lookat_mode: str,
    max_disparity: float | None,
    max_zoom: float | None,
    distance_m: float | None,
    num_steps: int,
    num_repeats: int,
    low_pass_filter_eps: float,
    cuda_device: int,
    target_triangles: int | None,
    progress: bool,
    verbose: bool,
) -> None:
    """Extract a colored triangle mesh from a SHARP Gaussians PLY file."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if not torch.cuda.is_available():
        LOGGER.error("Meshing requires CUDA.")
        raise SystemExit(1)
    try:
        gsplat.require_gsplat_cuda_extension()
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc

    base_params = dataclasses.replace(camera.TrajectoryParams(), num_steps=num_steps, num_repeats=num_repeats)
    params = dataclasses.replace(
        base_params,
        type=trajectory_type.lower(),
        lookat_mode=lookat_mode.lower(),
        max_disparity=max_disparity if max_disparity is not None else base_params.max_disparity,
        max_zoom=max_zoom if max_zoom is not None else base_params.max_zoom,
        distance_m=distance_m if distance_m is not None else base_params.distance_m,
    )

    options = MeshFromGaussiansOptions(
        max_render_dim=max_render_dim,
        depth_trunc=depth_trunc,
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        low_pass_filter_eps=low_pass_filter_eps,
        cuda_device=cuda_device,
    )

    mesh = gaussians_ply_to_mesh_tsdf(
        input_path, trajectory_params=params, options=options, progress=progress
    )

    if target_triangles is not None and target_triangles > 0:
        try:
            mesh = mesh.simplify_quadric_decimation(int(target_triangles))
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
        except Exception as exc:
            LOGGER.warning("Mesh simplification failed: %s", exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import open3d as o3d  # type: ignore[import-not-found]

        ok = o3d.io.write_triangle_mesh(str(output_path), mesh, write_vertex_colors=True)
    except Exception as exc:
        raise RuntimeError("Failed to write mesh. Ensure Open3D is installed.") from exc

    if not ok:
        raise RuntimeError(f"Failed to write mesh to {output_path}")
    LOGGER.info("Wrote mesh to %s", output_path)

