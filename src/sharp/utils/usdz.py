"""USDZ export helpers.

This uses a minimal USD ASCII (usda) writer plus ZIP packaging. It avoids
requiring the OpenUSD runtime (pxr) so it can run in lightweight environments.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData  # type: ignore[import-not-found]


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray  # (N,3) float32
    faces: np.ndarray  # (M,3) int32
    vertex_colors: np.ndarray | None  # (N,3) float32 [0,1]
    vertex_normals: np.ndarray | None  # (N,3) float32


def _fmt_f(x: float) -> str:
    # Keep files reasonably small while staying stable for USD parsing.
    return f"{float(x):.6g}"


def _fmt_vec3(v: np.ndarray) -> str:
    return f"({_fmt_f(v[0])}, {_fmt_f(v[1])}, {_fmt_f(v[2])})"


def _read_mesh_ply(path: Path) -> MeshData:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError("PLY has no 'vertex' element.")
    v = ply["vertex"].data

    for req in ("x", "y", "z"):
        if req not in ply["vertex"]:
            raise ValueError(f"PLY vertex is missing '{req}'.")

    vertices = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32, copy=False)

    normals = None
    if all(k in ply["vertex"] for k in ("nx", "ny", "nz")):
        normals = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32, copy=False)

    colors = None
    if all(k in ply["vertex"] for k in ("red", "green", "blue")):
        rgb_u8 = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32, copy=False)
        colors = (rgb_u8 / 255.0).clip(0.0, 1.0).astype(np.float32, copy=False)
    elif all(k in ply["vertex"] for k in ("r", "g", "b")):
        rgb_u8 = np.stack([v["r"], v["g"], v["b"]], axis=1).astype(np.float32, copy=False)
        colors = (rgb_u8 / 255.0).clip(0.0, 1.0).astype(np.float32, copy=False)

    if "face" not in ply:
        raise ValueError("PLY has no 'face' element.")
    f = ply["face"].data
    if "vertex_indices" not in ply["face"]:
        raise ValueError("PLY face is missing 'vertex_indices'.")

    # `vertex_indices` is typically a list per face. Require triangles for now.
    face_lists = f["vertex_indices"]
    if len(face_lists) == 0:
        raise ValueError("PLY has 0 faces.")
    tri = []
    for idxs in face_lists:
        if len(idxs) != 3:
            raise ValueError("Only triangle meshes are supported for USDZ export.")
        tri.append([int(idxs[0]), int(idxs[1]), int(idxs[2])])
    faces = np.asarray(tri, dtype=np.int32)

    return MeshData(vertices=vertices, faces=faces, vertex_colors=colors, vertex_normals=normals)


def mesh_to_usda(
    mesh: MeshData,
    *,
    root_name: str = "Root",
    mesh_name: str = "Mesh",
    material_name: str = "Mat",
) -> str:
    """Serialize a mesh (with optional vertex colors) into a minimal usda stage."""
    vertices = mesh.vertices
    faces = mesh.faces

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N,3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M,3)")

    normals = mesh.vertex_normals
    if normals is not None and (normals.shape != vertices.shape):
        raise ValueError("vertex_normals must have shape (N,3)")

    colors = mesh.vertex_colors
    if colors is not None and (colors.shape != vertices.shape):
        raise ValueError("vertex_colors must have shape (N,3)")

    face_counts = np.full((faces.shape[0],), 3, dtype=np.int32)
    face_indices = faces.reshape(-1).astype(np.int32, copy=False)

    s = io.StringIO()
    s.write("#usda 1.0\n")
    s.write("(\n")
    s.write('    metersPerUnit = 1\n')
    s.write('    upAxis = "Y"\n')
    s.write(")\n\n")
    s.write(f'def Xform "{root_name}" {{\n')
    s.write(f'    def Mesh "{mesh_name}" (\n')
    s.write('        prepend apiSchemas = ["MaterialBindingAPI"]\n')
    s.write("    ) {\n")
    s.write("        uniform token subdivisionScheme = \"none\"\n")

    # Geometry
    s.write("        int[] faceVertexCounts = [")
    s.write(", ".join(str(int(x)) for x in face_counts))
    s.write("]\n")

    s.write("        int[] faceVertexIndices = [")
    s.write(", ".join(str(int(x)) for x in face_indices))
    s.write("]\n")

    s.write("        point3f[] points = [\n")
    for v in vertices:
        s.write(f"            {_fmt_vec3(v)},\n")
    s.write("        ]\n")

    if normals is not None:
        s.write("        normal3f[] normals = [\n")
        for n in normals:
            s.write(f"            {_fmt_vec3(n)},\n")
        s.write('        ] (interpolation = "vertex")\n')

    if colors is not None:
        s.write("        color3f[] primvars:displayColor = [\n")
        for c in colors:
            s.write(f"            {_fmt_vec3(c)},\n")
        s.write('        ] (interpolation = "vertex")\n')

    # Bind a simple preview material that reads the vertex color primvar.
    s.write(f'        rel material:binding = </{root_name}/Looks/{material_name}>\n')
    s.write("    }\n\n")

    s.write('    def Scope "Looks" {\n')
    s.write(f'        def Material "{material_name}" {{\n')
    s.write(
        f'            token outputs:surface.connect = </{root_name}/Looks/{material_name}/PreviewSurface.outputs:surface>\n'
    )
    s.write(f'            def Shader "PreviewSurface" {{\n')
    s.write('                uniform token info:id = "UsdPreviewSurface"\n')
    if colors is not None:
        s.write(
            f'                color3f inputs:diffuseColor.connect = </{root_name}/Looks/{material_name}/ColorReader.outputs:result>\n'
        )
    else:
        s.write("                color3f inputs:diffuseColor = (0.7, 0.7, 0.7)\n")
    s.write("                float inputs:roughness = 0.5\n")
    s.write("                float inputs:metallic = 0\n")
    s.write("                token outputs:surface\n")
    s.write("            }\n")
    if colors is not None:
        s.write(f'            def Shader "ColorReader" {{\n')
        s.write('                uniform token info:id = "UsdPrimvarReader_color3f"\n')
        s.write('                token inputs:varname = "displayColor"\n')
        s.write("                color3f outputs:result\n")
        s.write("            }\n")
    s.write("        }\n")
    s.write("    }\n")
    s.write("}\n")

    return s.getvalue()


def write_usdz(usdz_path: Path, *, usda_text: str, usda_name: str = "scene.usda") -> None:
    """Create a USDZ package containing a single usda stage."""
    usdz_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(usdz_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(usda_name, usda_text)


def mesh_ply_to_usdz(
    input_mesh_ply: Path,
    *,
    output_usdz: Path,
    root_name: str = "Root",
    mesh_name: str = "Mesh",
) -> None:
    """Convert a mesh PLY (with vertex colors) to a USDZ package."""
    mesh = _read_mesh_ply(input_mesh_ply)
    usda = mesh_to_usda(mesh, root_name=root_name, mesh_name=mesh_name)
    write_usdz(output_usdz, usda_text=usda)

