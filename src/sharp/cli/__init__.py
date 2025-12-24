"""Command-line-interface to run SHARP model.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import click

from . import mesh, photo_to_viewer, predict, render, usdz


@click.group()
def main_cli():
    """Run inference for SHARP model."""
    pass


main_cli.add_command(predict.predict_cli, "predict")
main_cli.add_command(render.render_cli, "render")
main_cli.add_command(mesh.mesh_cli, "mesh")
main_cli.add_command(usdz.usdz_cli, "usdz")
main_cli.add_command(photo_to_viewer.photo_to_viewer_cli, "photo-to-viewer")
