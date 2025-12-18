"""Utilities for converting Gaussians to triangle meshes.

This module is intentionally optional-dependency friendly: it only imports Open3D
when meshing is actually invoked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sharp.utils import camera, gsplat
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, load_ply

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    import open3d as o3d


@dataclass(frozen=True)
class MeshFromGaussiansOptions:
    """Options for multi-view TSDF meshing from 3D Gaussians."""

    max_render_dim: int = 640
    depth_trunc: float = 10.0
    voxel_length: float = 0.01
    sdf_trunc: float = 0.04
    low_pass_filter_eps: float = 0.0
    cuda_device: int = 0


def _import_open3d() -> Any:
    try:
        import open3d as o3d  # type: ignore[import-not-found]

        return o3d
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Meshing requires Open3D. Install it with `pip install open3d` "
            "(or add it to your environment) and try again."
        ) from exc


def _scaled_camera_metadata(
    metadata: SceneMetaData, *, max_dim: int
) -> tuple[float, tuple[int, int], float]:
    """Return (scaled_f_px, scaled_resolution, scale_factor)."""
    (src_w, src_h) = metadata.resolution_px
    if max_dim <= 0:
        raise ValueError("max_dim must be > 0")
    scale = min(1.0, float(max_dim) / float(max(src_w, src_h)))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    f_px = float(metadata.focal_length_px) * scale
    return f_px, (dst_w, dst_h), scale


@torch.no_grad()
def gaussians_to_mesh_tsdf(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    *,
    trajectory_params: camera.TrajectoryParams,
    options: MeshFromGaussiansOptions,
    progress: bool = True,
) -> "o3d.geometry.TriangleMesh":
    """Convert Gaussians to a colored triangle mesh using multi-view TSDF fusion."""
    o3d = _import_open3d()

    if not torch.cuda.is_available():
        raise RuntimeError("Meshing requires CUDA, but torch.cuda is not available.")
    gsplat.require_gsplat_cuda_extension()

    device = torch.device(f"cuda:{int(options.cuda_device)}")
    gaussians_device = gaussians.to(device)

    f_px, resolution_px, _scale = _scaled_camera_metadata(metadata, max_dim=options.max_render_dim)
    (width, height) = resolution_px
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

    camera_model = camera.create_camera_model(
        gaussians, intrinsics, resolution_px=resolution_px, lookat_mode=trajectory_params.lookat_mode
    )
    trajectory = camera.create_eye_trajectory(
        gaussians, trajectory_params, resolution_px=resolution_px, f_px=f_px
    )
    total = len(trajectory)
    LOGGER.info("Meshing from %d views (TSDF).", total)

    renderer = gsplat.GSplatRenderer(
        color_space=metadata.color_space,
        low_pass_filter_eps=float(options.low_pass_filter_eps),
    )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(options.voxel_length),
        sdf_trunc=float(options.sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    intrinsic_o3d = None
    for index, eye_position in enumerate(trajectory, start=1):
        camera_info = camera_model.compute(eye_position)
        rendering = renderer(
            gaussians_device,
            extrinsics=camera_info.extrinsics[None].to(device),
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )

        # GSplatRenderer outputs in sRGB in [0,1].
        color_u8 = (
            rendering.color[0].clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).to(torch.uint8)
        )
        depth_m = rendering.depth[0].squeeze(0).clamp(min=0.0)

        # Open3D requires C-contiguous buffers for Image initialization.
        color_np = np.ascontiguousarray(color_u8.detach().cpu().numpy())
        depth_np = np.ascontiguousarray(depth_m.detach().cpu().numpy(), dtype=np.float32)

        if intrinsic_o3d is None:
            fx = float(camera_info.intrinsics[0, 0].item())
            fy = float(camera_info.intrinsics[1, 1].item())
            cx = float(camera_info.intrinsics[0, 2].item())
            cy = float(camera_info.intrinsics[1, 2].item())
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
                camera_info.width, camera_info.height, fx, fy, cx, cy
            )

        color_o3d = o3d.geometry.Image(color_np)
        depth_o3d = o3d.geometry.Image(depth_np)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=float(options.depth_trunc),
            convert_rgb_to_intensity=False,
        )

        # camera_info.extrinsics is world->camera; Open3D expects camera->world.
        extr_w2c = camera_info.extrinsics.detach().cpu().numpy().astype(np.float64, copy=False)
        extr_c2w = np.linalg.inv(extr_w2c)

        volume.integrate(rgbd, intrinsic_o3d, extr_c2w)

        if progress and (index == 1 or index == total or index % 10 == 0):
            LOGGER.info("Integrated %d/%d views", index, total)

    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def gaussians_ply_to_mesh_tsdf(
    input_path: Path,
    *,
    trajectory_params: camera.TrajectoryParams,
    options: MeshFromGaussiansOptions,
    progress: bool = True,
) -> "o3d.geometry.TriangleMesh":
    """Load Gaussians PLY and convert to mesh via TSDF."""
    gaussians, metadata = load_ply(input_path)
    return gaussians_to_mesh_tsdf(
        gaussians,
        metadata,
        trajectory_params=trajectory_params,
        options=options,
        progress=progress,
    )
