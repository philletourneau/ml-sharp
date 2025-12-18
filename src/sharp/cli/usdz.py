"""Contains `sharp usdz` CLI implementation."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import click
import numpy as np
import torch
from plyfile import PlyData  # type: ignore[import-not-found]

from sharp.utils import camera, gsplat
from sharp.utils import logging as logging_utils
from sharp.utils.meshing import MeshFromGaussiansOptions, gaussians_ply_to_mesh_tsdf
from sharp.utils.usdz import CoordinateSystem, MeshData, mesh_ply_to_usdz, mesh_to_usda, write_usdz

LOGGER = logging.getLogger(__name__)


def _is_gaussians_ply(path: Path) -> bool:
    """Return True if the PLY looks like a SHARP Gaussians PLY (not a triangle mesh PLY)."""
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        return False
    vertex = ply["vertex"]
    return all(k in vertex for k in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "rot_0"))


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a mesh PLY (preferred) or a SHARP Gaussians PLY.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to write the output .usdz file.",
    required=True,
)
@click.option(
    "--root-name",
    type=str,
    default="Root",
    show_default=True,
    help="USD root prim name.",
)
@click.option(
    "--mesh-name",
    type=str,
    default="Mesh",
    show_default=True,
    help="USD mesh prim name.",
)
@click.option(
    "--coordinate-system",
    type=click.Choice(["usd", "sharp"], case_sensitive=False),
    default="usd",
    show_default=True,
    help="Coordinate system for the USD stage. 'usd' converts from SHARP (x right, y down, z forward) to a USD-friendly right-handed Y-up space.",
)
@click.option(
    "--mesh-output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional path to also write the intermediate mesh PLY when input is Gaussians.",
)
# Meshing options (only used when input is Gaussians PLY).
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
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Print periodic progress while integrating views (Gaussians input only).",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def usdz_cli(
    input_path: Path,
    output_path: Path,
    root_name: str,
    mesh_name: str,
    coordinate_system: str,
    mesh_output: Path | None,
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
    progress: bool,
    verbose: bool,
) -> None:
    """Export a mesh to USDZ (RealityKit-friendly)."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    # Fast path: mesh PLY in -> USDZ out.
    if not _is_gaussians_ply(input_path):
        mesh_ply_to_usdz(
            input_path,
            output_usdz=output_path,
            root_name=root_name,
            mesh_name=mesh_name,
            coordinate_system=coordinate_system.lower(),  # type: ignore[arg-type]
        )
        LOGGER.info("Wrote USDZ to %s", output_path)
        return

    # Gaussians PLY -> mesh -> USDZ
    if not torch.cuda.is_available():
        LOGGER.error("USDZ export from Gaussians requires CUDA.")
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

    if mesh_output is not None:
        mesh_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            import open3d as o3d  # type: ignore[import-not-found]

            o3d.io.write_triangle_mesh(str(mesh_output), mesh, write_vertex_colors=True)
            LOGGER.info("Wrote intermediate mesh to %s", mesh_output)
        except Exception as exc:
            LOGGER.warning("Failed to write intermediate mesh: %s", exc)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = None
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    normals = None
    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    usda = mesh_to_usda(
        MeshData(vertices=vertices, faces=faces, vertex_colors=colors, vertex_normals=normals),
        root_name=root_name,
        mesh_name=mesh_name,
        coordinate_system=coordinate_system.lower(),  # type: ignore[arg-type]
    )
    write_usdz(output_path, usda_text=usda)
    LOGGER.info("Wrote USDZ to %s", output_path)
