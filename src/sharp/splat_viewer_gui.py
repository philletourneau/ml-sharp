"""Windows GUI for generating a PlayCanvas HTML splat viewer using splat-transform.

This app wraps the `@playcanvas/splat-transform` CLI and produces a single-page HTML viewer.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import queue
import random
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MOTION_PRESET_FPS = 6
MOTION_AMPLITUDE_SCALE = 0.65


@dataclass(frozen=True)
class RunConfig:
    input_path: Path
    output_html: Path
    viewer_settings: Path | None
    unbundled: bool
    overwrite: bool
    enable_aa: bool
    enable_msaa: bool
    match_gsplat_camera: bool
    gpu: str | None  # "cpu" or an integer string
    iterations: int | None
    extra_args: str


def _apply_viewer_customizations(
    output_html: Path,
    *,
    unbundled: bool,
    enable_aa: bool,
    enable_msaa: bool,
    motion_presets: dict[str, object] | None,
) -> None:
    """Post-process SuperSplat Viewer output for SHARP-style defaults and motion controls."""

    def patch_text_file(path: Path, patcher: Callable[[str], str]) -> None:
        text = path.read_text(encoding="utf-8")
        patched = patcher(text)
        if patched != text:
            path.write_text(patched, encoding="utf-8")

    presets_payload = motion_presets or {"version": 1, "fps": MOTION_PRESET_FPS, "presets": []}
    presets_json = json.dumps(presets_payload, separators=(",", ":"))

    def patch_viewer_html(text: str) -> str:
        patched = text
        aa_value = "true" if enable_aa else "false"
        aa_pattern = r"aa:\s*(?:url\.searchParams\.has\('aa'\)|true|false)"
        if re.search(aa_pattern, patched):
            patched = re.sub(aa_pattern, f"aa: {aa_value}", patched, count=1)
        else:
            raise RuntimeError("Could not patch AA default (expected aa setting not found).")

        noanim_pattern = r"noanim:\s*url\.searchParams\.has\('noanim'\)"
        if re.search(noanim_pattern, patched):
            patched = re.sub(
                noanim_pattern,
                "noanim: url.searchParams.has('noanim') || !url.searchParams.has('anim')",
                patched,
                count=1,
            )
        else:
            raise RuntimeError("Could not patch noanim default (expected noanim setting not found).")

        msaa_value = "true" if enable_msaa else "false"
        msaa_pattern = r'antialias="(?:true|false)"'
        if re.search(msaa_pattern, patched):
            patched = re.sub(msaa_pattern, f'antialias="{msaa_value}"', patched, count=1)
        else:
            raise RuntimeError("Could not patch MSAA default (expected antialias attribute not found).")

        if "sharp-motion-style" not in patched:
            style_block = (
                "<style id=\"sharp-motion-style\">\n"
                "#settingsPanel .settingsLabel {\n"
                "  cursor: default;\n"
                "  color: #E0DCDD;\n"
                "}\n"
                "#settingsPanel select,\n"
                "#settingsPanel input[type=range] {\n"
                "  flex-grow: 1;\n"
                "  min-width: 0;\n"
                "  padding: 6px 8px;\n"
                "  border: 0;\n"
                "  color: #E0DCDD;\n"
                "  background-color: #141414;\n"
                "}\n"
                "#settingsPanel input[type=range] {\n"
                "  padding: 0;\n"
                "}\n"
                "#settingsPanel #motionSpeedValue {\n"
                "  padding: 8px 6px;\n"
                "  min-width: 48px;\n"
                "  text-align: right;\n"
                "  color: #E0DCDD;\n"
                "}\n"
                "#settingsPanel #cameraDebugInfo {\n"
                "  width: 100%;\n"
                "  white-space: pre;\n"
                "  font-family: Consolas, \"Courier New\", monospace;\n"
                "  font-size: 11px;\n"
                "  line-height: 1.3;\n"
                "  color: #CFCFCF;\n"
                "  background-color: #1f1f1f;\n"
                "  border-radius: 4px;\n"
                "  padding: 6px;\n"
                "}\n"
                "</style>\n"
            )
            if "</head>" not in patched:
                raise RuntimeError("Could not inject motion styles (missing </head>).")
            patched = patched.replace("</head>", style_block + "</head>", 1)

        if "id=\"motionPreset\"" not in patched:
            controls_block = (
                "                <div class=\"divider\"></div>\n"
                "                <div class=\"settingsRow\">\n"
                "                    <div class=\"settingsLabel\">Motion</div>\n"
                "                    <select id=\"motionPreset\"></select>\n"
                "                </div>\n"
                "                <div class=\"settingsRow\">\n"
                "                    <button id=\"motionPlay\">Play Motion</button>\n"
                "                    <button id=\"motionStop\">Stop Motion</button>\n"
                "                </div>\n"
                "                <div class=\"settingsRow\">\n"
                "                    <div class=\"settingsLabel\">Speed</div>\n"
                "                    <input id=\"motionSpeed\" type=\"range\" min=\"0.25\" max=\"2\" step=\"0.05\" value=\"1\">\n"
                "                    <div id=\"motionSpeedValue\">1.00x</div>\n"
                "                </div>\n"
                "                <div class=\"divider\"></div>\n"
                "                <div class=\"settingsRow\">\n"
                "                    <button id=\"cameraDebug\">Camera Debug</button>\n"
                "                </div>\n"
                "                <div class=\"settingsRow\">\n"
                "                    <div id=\"cameraDebugInfo\" class=\"hidden\"></div>\n"
                "                </div>\n"
            )
            marker = (
                "                <div class=\"settingsRow\">\n"
                "                    <button id=\"frame\">Frame</button>\n"
                "                    <button id=\"reset\">Reset</button>\n"
                "                </div>"
            )
            if marker not in patched:
                raise RuntimeError("Could not inject motion controls (settings buttons not found).")
            patched = patched.replace(marker, marker + "\n" + controls_block, 1)

        if "window.sharpMotionPresets" not in patched:
            sse_pattern = r"(window\.sse\s*=\s*\{.*?\};)"
            if not re.search(sse_pattern, patched, flags=re.S):
                raise RuntimeError("Could not inject motion presets (window.sse not found).")
            patched = re.sub(
                sse_pattern,
                r"\1\n            window.sharpMotionPresets = " + presets_json + ";",
                patched,
                count=1,
                flags=re.S,
            )

        return patched

    if output_html.is_file():
        patch_text_file(output_html, patch_viewer_html)

    js_path = output_html if not unbundled else output_html.parent / "index.js"
    if not js_path.is_file():
        raise RuntimeError(f"Expected viewer script not found: {js_path}")

    def patch_viewer_js(text: str) -> str:
        patched = text

        if "buildGsplatMotionTrack" not in patched:
            helpers = (
                "const buildGsplatMotionTrack = (preset, baseCamera, fps = 6) => {\n"
                "    if (!preset || !baseCamera) {\n"
                "        return null;\n"
                "    }\n"
                "    const numSteps = Math.max(2, Math.trunc(preset.numSteps || 60));\n"
                "    const numRepeats = Math.max(1, Math.trunc(preset.numRepeats || 1));\n"
                "    const totalSteps = Math.max(2, numSteps * numRepeats);\n"
                "    const duration = totalSteps / fps;\n"
                "    const times = new Array(totalSteps);\n"
                "    const positions = [];\n"
                "    const targets = [];\n"
                "    const fovs = new Array(totalSteps).fill(baseCamera.fov);\n"
                "    const basePos = new Vec3(baseCamera.position);\n"
                "    const baseTarget = new Vec3(baseCamera.target);\n"
                "    const forward = new Vec3();\n"
                "    forward.sub2(baseTarget, basePos);\n"
                "    const forwardDist = forward.length() || 1;\n"
                "    forward.mulScalar(1 / forwardDist);\n"
                "    const maxOffset = preset.maxOffset || [0, 0, 0];\n"
                "    const distance = preset.distance || 0;\n"
                "    const type = preset.type || 'rotate_forward';\n"
                "    const lookat = preset.lookat || 'point';\n"
                "    const halfSteps = Math.floor(totalSteps / 2);\n"
                "    const horizontalRatio = totalSteps > 0 ? (halfSteps / totalSteps) : 0.5;\n"
                "    const pos = new Vec3();\n"
                "    const target = new Vec3();\n"
                "    const twoPi = Math.PI * 2;\n"
                "\n"
                "    const samplePosition = (t) => {\n"
                "        const phase = t * numRepeats;\n"
                "        if (type === 'swipe') {\n"
                "            const localT = phase - Math.floor(phase);\n"
                "            pos.set(\n"
                "                -maxOffset[0] + (maxOffset[0] * 2 * localT),\n"
                "                0,\n"
                "                distance\n"
                "            );\n"
                "            return;\n"
                "        }\n"
                "        if (type === 'shake') {\n"
                "            if (t < horizontalRatio) {\n"
                "                const localT = horizontalRatio > 0 ? (t / horizontalRatio) : 0;\n"
                "                const angle = twoPi * (localT * numRepeats);\n"
                "                pos.set(maxOffset[0] * Math.sin(angle), 0, distance);\n"
                "            }\n"
                "            else {\n"
                "                const denom = 1 - horizontalRatio;\n"
                "                const localT = denom > 0 ? ((t - horizontalRatio) / denom) : 0;\n"
                "                const angle = twoPi * (localT * numRepeats);\n"
                "                pos.set(0, maxOffset[1] * Math.sin(angle), distance);\n"
                "            }\n"
                "            return;\n"
                "        }\n"
                "        const angle = twoPi * phase;\n"
                "        if (type === 'rotate') {\n"
                "            pos.set(\n"
                "                maxOffset[0] * Math.sin(angle),\n"
                "                maxOffset[1] * Math.cos(angle),\n"
                "                distance\n"
                "            );\n"
                "            return;\n"
                "        }\n"
                "        pos.set(\n"
                "            maxOffset[0] * Math.sin(angle),\n"
                "            0,\n"
                "            distance + maxOffset[2] * (1 - Math.cos(angle)) * 0.5\n"
                "        );\n"
                "    };\n"
                "\n"
                "    for (let i = 0; i < totalSteps; i++) {\n"
                "        const t = totalSteps > 1 ? i / (totalSteps - 1) : 0;\n"
                "        times[i] = t * duration;\n"
                "        samplePosition(t);\n"
                "        pos.add(basePos);\n"
                "        positions.push(pos.x, pos.y, pos.z);\n"
                "        if (lookat === 'ahead') {\n"
                "            target.copy(forward).mulScalar(forwardDist).add(pos);\n"
                "        }\n"
                "        else {\n"
                "            target.copy(baseTarget);\n"
                "        }\n"
                "        targets.push(target.x, target.y, target.z);\n"
                "    }\n"
                "    return {\n"
                "        name: preset.id || preset.type || 'gsplat',\n"
                "        duration,\n"
                "        frameRate: 1,\n"
                "        loopMode: 'repeat',\n"
                "        interpolation: 'spline',\n"
                "        smoothness: 1,\n"
                "        keyframes: {\n"
                "            times,\n"
                "            values: {\n"
                "                position: positions,\n"
                "                target: targets,\n"
                "                fov: fovs\n"
                "            }\n"
                "        }\n"
                "    };\n"
                "};\n"
                "\n"
            )
            marker = "class CubicSpline"
            if marker not in patched:
                raise RuntimeError("Could not inject motion helpers (CubicSpline not found).")
            patched = patched.replace(marker, helpers + marker, 1)

        if "animationSpeed" not in patched:
            speed_anchor = "animationPaused: true,"
            if speed_anchor not in patched:
                raise RuntimeError("Could not patch animation speed (state anchor not found).")
            patched = patched.replace(
                speed_anchor,
                speed_anchor + "\n        animationSpeed: 1,",
                1,
            )

        dt_anchor = "const dt = state.cameraMode === 'anim' && state.animationPaused ? 0 : deltaTime;"
        if dt_anchor in patched:
            patched = patched.replace(
                dt_anchor,
                "const dt = state.cameraMode === 'anim'\n"
                "            ? (state.animationPaused ? 0 : deltaTime * (state.animationSpeed || 1))\n"
                "            : deltaTime;",
                1,
            )
        else:
            raise RuntimeError("Could not patch animation speed (dt expression not found).")

        if "setAnimTrack" not in patched:
            anim_anchor = (
                "state.animationDuration = controllers.anim ? controllers.anim.animState.cursor.duration : 0;"
            )
            if anim_anchor not in patched:
                raise RuntimeError("Could not inject anim track setter (duration anchor not found).")
            patched = patched.replace(
                anim_anchor,
                anim_anchor
                + "\n        this.setAnimTrack = (track) => {\n"
                "            if (!track) {\n"
                "                return;\n"
                "            }\n"
                "            controllers.anim = new AnimController(track);\n"
                "            state.hasAnimation = true;\n"
                "            state.animationDuration = controllers.anim.animState.cursor.duration;\n"
                "            state.animationTime = 0;\n"
                "        };",
                1,
            )

        if "global.cameraManager" not in patched:
            camera_anchor = (
                "this.cameraManager = new CameraManager(global, sceneBound);\n"
                "            applyCamera(this.cameraManager.camera);"
            )
            if camera_anchor not in patched:
                raise RuntimeError("Could not expose camera manager (anchor not found).")
            patched = patched.replace(
                camera_anchor,
                "this.cameraManager = new CameraManager(global, sceneBound);\n"
                "            global.cameraManager = this.cameraManager;\n"
                "            applyCamera(this.cameraManager.camera);",
                1,
            )

        dom_block = ""
        dom_match = re.search(r"const\\s+dom\\s*=\\s*\\[(.*?)\\]\\s*\\.reduce", patched, flags=re.S)
        if dom_match:
            dom_block = dom_match.group(1)
        if "motionPreset" not in dom_block:
            dom_anchor = "'reset', 'frame',"
            if dom_anchor not in patched:
                raise RuntimeError("Could not add motion UI bindings (dom anchor not found).")
            patched = patched.replace(
                dom_anchor,
                "'reset', 'frame',\n"
                "        'motionPreset', 'motionPlay', 'motionStop', 'motionSpeed', 'motionSpeedValue',\n"
                "        'cameraDebug', 'cameraDebugInfo',",
                1,
            )

        if "motionPresets" not in patched:
            frame_anchor = (
                "    dom.frame.addEventListener('click', (event) => {\n"
                "        events.fire('inputEvent', 'frame', event);\n"
                "    });"
            )
            if frame_anchor not in patched:
                raise RuntimeError("Could not add motion UI handlers (frame handler not found).")
            motion_ui = (
                "\n"
                "    const motionPresets = (window.sharpMotionPresets && window.sharpMotionPresets.presets) || [];\n"
                "    const motionFps = (window.sharpMotionPresets && window.sharpMotionPresets.fps) || 6;\n"
                "    let motionPrevMode = 'orbit';\n"
                "    const motionControlsReady = dom.motionPreset && dom.motionPlay && dom.motionStop && dom.motionSpeed && dom.motionSpeedValue;\n"
                "    if (motionControlsReady) {\n"
                "        if (motionPresets.length > 0) {\n"
                "            dom.motionPreset.textContent = '';\n"
                "            motionPresets.forEach((preset, index) => {\n"
                "                const option = document.createElement('option');\n"
                "                option.value = preset.id || preset.type || `preset_${index}`;\n"
                "                option.textContent = preset.label || preset.type || `Preset ${index + 1}`;\n"
                "                dom.motionPreset.appendChild(option);\n"
                "            });\n"
                "            const getSelectedPreset = () => {\n"
                "                const selectedId = dom.motionPreset.value;\n"
                "                return motionPresets.find((preset) => (preset.id || preset.type) === selectedId) || motionPresets[0];\n"
                "            };\n"
                "            const applyMotionPreset = () => {\n"
                "                const baseCamera = global.settings.cameras && global.settings.cameras[0]\n"
                "                    ? global.settings.cameras[0].initial\n"
                "                    : null;\n"
                "                if (!baseCamera || !global.cameraManager || !global.cameraManager.setAnimTrack) {\n"
                "                    return;\n"
                "                }\n"
                "                const preset = getSelectedPreset();\n"
                "                const track = buildGsplatMotionTrack(preset, baseCamera, motionFps);\n"
                "                if (track) {\n"
                "                    global.cameraManager.setAnimTrack(track);\n"
                "                }\n"
                "            };\n"
                "            dom.motionPreset.addEventListener('change', () => {\n"
                "                applyMotionPreset();\n"
                "            });\n"
                "            dom.motionPlay.addEventListener('click', () => {\n"
                "                motionPrevMode = state.cameraMode;\n"
                "                applyMotionPreset();\n"
                "                state.cameraMode = 'anim';\n"
                "                state.animationPaused = false;\n"
                "            });\n"
                "            dom.motionStop.addEventListener('click', () => {\n"
                "                state.animationPaused = true;\n"
                "                if (motionPrevMode) {\n"
                "                    state.cameraMode = motionPrevMode;\n"
                "                }\n"
                "            });\n"
                "            const updateSpeedLabel = (value) => {\n"
                "                const speed = Math.max(0.25, Math.min(2, value));\n"
                "                dom.motionSpeedValue.textContent = `${speed.toFixed(2)}x`;\n"
                "                return speed;\n"
                "            };\n"
                "            const updateSpeed = () => {\n"
                "                const raw = parseFloat(dom.motionSpeed.value);\n"
                "                state.animationSpeed = updateSpeedLabel(isFinite(raw) ? raw : 1);\n"
                "            };\n"
                "            dom.motionSpeed.addEventListener('input', updateSpeed);\n"
                "            events.on('animationSpeed:changed', (value) => {\n"
                "                if (document.activeElement !== dom.motionSpeed) {\n"
                "                    dom.motionSpeed.value = value.toString();\n"
                "                }\n"
                "                updateSpeedLabel(value);\n"
                "            });\n"
                "            dom.motionSpeed.value = (state.animationSpeed || 1).toString();\n"
                "            updateSpeedLabel(state.animationSpeed || 1);\n"
                "            events.on('firstFrame', () => {\n"
                "                applyMotionPreset();\n"
                "            });\n"
                "        }\n"
                "        else {\n"
                "            dom.motionPreset.disabled = true;\n"
                "            dom.motionPlay.disabled = true;\n"
                "            dom.motionStop.disabled = true;\n"
                "            dom.motionSpeed.disabled = true;\n"
                "            dom.motionSpeedValue.textContent = '--';\n"
                "        }\n"
                "    }\n"
                "    if (dom.cameraDebug && dom.cameraDebugInfo) {\n"
                "        let debugActive = false;\n"
                "        let debugHandle = 0;\n"
                "        const debugTarget = new Vec3();\n"
                "        const updateDebug = () => {\n"
                "            if (!debugActive) {\n"
                "                return;\n"
                "            }\n"
                "            const cam = global.cameraManager ? global.cameraManager.camera : null;\n"
                "            if (cam) {\n"
                "                cam.calcFocusPoint(debugTarget);\n"
                "                dom.cameraDebugInfo.textContent =\n"
                "                    `pos: ${cam.position.x.toFixed(3)}, ${cam.position.y.toFixed(3)}, ${cam.position.z.toFixed(3)}\\n` +\n"
                "                    `rot: ${cam.angles.x.toFixed(2)}, ${cam.angles.y.toFixed(2)}, ${cam.angles.z.toFixed(2)}\\n` +\n"
                "                    `fov: ${cam.fov.toFixed(2)}  dist: ${cam.distance.toFixed(3)}\\n` +\n"
                "                    `target: ${debugTarget.x.toFixed(3)}, ${debugTarget.y.toFixed(3)}, ${debugTarget.z.toFixed(3)}`;\n"
                "            }\n"
                "            else {\n"
                "                dom.cameraDebugInfo.textContent = 'Camera not ready';\n"
                "            }\n"
                "            debugHandle = requestAnimationFrame(updateDebug);\n"
                "        };\n"
                "        dom.cameraDebug.addEventListener('click', () => {\n"
                "            debugActive = !debugActive;\n"
                "            dom.cameraDebugInfo.classList.toggle('hidden', !debugActive);\n"
                "            dom.cameraDebug.textContent = debugActive ? 'Hide Camera Debug' : 'Camera Debug';\n"
                "            if (debugActive) {\n"
                "                updateDebug();\n"
                "            }\n"
                "            else if (debugHandle) {\n"
                "                cancelAnimationFrame(debugHandle);\n"
                "                debugHandle = 0;\n"
                "            }\n"
                "        });\n"
                "    }"
            )
            patched = patched.replace(frame_anchor, frame_anchor + motion_ui, 1)

        tooltip_anchor = "tooltip.register(dom.exitFullscreen, 'Fullscreen', 'top');"
        if tooltip_anchor in patched and "Play Motion" not in patched:
            patched = patched.replace(
                tooltip_anchor,
                tooltip_anchor
                + "\n    if (dom.motionPlay) {\n"
                "        tooltip.register(dom.motionPlay, 'Play Motion', 'bottom');\n"
                "        tooltip.register(dom.motionStop, 'Stop Motion', 'bottom');\n"
                "    }\n"
                "    if (dom.cameraDebug) {\n"
                "        tooltip.register(dom.cameraDebug, 'Camera Debug', 'bottom');\n"
                "    }",
                1,
            )

        return patched

    patch_text_file(js_path, patch_viewer_js)


_PLY_SCALAR_TYPE_SIZES: dict[str, int] = {
    "char": 1,
    "int8": 1,
    "uchar": 1,
    "uint8": 1,
    "short": 2,
    "int16": 2,
    "ushort": 2,
    "uint16": 2,
    "int": 4,
    "int32": 4,
    "uint": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}

_PLY_SCALAR_TYPE_STRUCT: dict[str, str] = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _parse_ply_header(path: Path) -> tuple[str, list[tuple[str, int, list[str]]], int]:
    """Parse enough of a PLY header to locate element boundaries.

    Returns:
        format_line: The `format ...` header line (without newline).
        elements: List of (name, count, property_lines).
        data_start: Byte offset where element data starts (after end_header newline).
    """
    with path.open("rb") as f:
        first = f.readline()
        if not first:
            raise RuntimeError("Empty file.")
        if first.strip() != b"ply":
            raise RuntimeError("Not a PLY file (missing 'ply' header).")

        format_line_b = f.readline()
        if not format_line_b:
            raise RuntimeError("Truncated PLY header (missing format line).")
        format_line = format_line_b.decode("ascii", errors="strict").strip()
        if not format_line.startswith("format "):
            raise RuntimeError("Invalid PLY header (missing format line).")

        elements: list[tuple[str, int, list[str]]] = []
        current_name: str | None = None
        current_count: int | None = None
        current_props: list[str] = []

        while True:
            line_b = f.readline()
            if not line_b:
                raise RuntimeError("Truncated PLY header (missing end_header).")
            line = line_b.decode("ascii", errors="strict").strip()
            if not line or line.startswith("comment"):
                continue
            if line == "end_header":
                if current_name is not None and current_count is not None:
                    elements.append((current_name, current_count, current_props))
                data_start = f.tell()
                return format_line, elements, data_start

            parts = line.split()
            if parts[0] == "element":
                if len(parts) != 3:
                    raise RuntimeError(f"Invalid element line: {line}")
                if current_name is not None and current_count is not None:
                    elements.append((current_name, current_count, current_props))
                current_name = parts[1]
                current_count = int(parts[2])
                current_props = []
            elif parts[0] == "property":
                if current_name is None:
                    continue
                current_props.append(line)


def _element_fixed_record_size(prop_lines: list[str]) -> int:
    size = 0
    for line in prop_lines:
        parts = line.split()
        if len(parts) < 3 or parts[0] != "property":
            continue
        if parts[1] == "list":
            raise RuntimeError(
                "PLY contains variable-length list properties; cannot strip metadata safely."
            )
        prop_type = parts[1].lower()
        if prop_type not in _PLY_SCALAR_TYPE_SIZES:
            raise RuntimeError(f"Unsupported PLY scalar type: {prop_type}")
        size += _PLY_SCALAR_TYPE_SIZES[prop_type]
    return size


def _element_single_property_type(prop_lines: list[str]) -> str | None:
    prop_types = []
    for line in prop_lines:
        parts = line.split()
        if len(parts) < 3 or parts[0] != "property":
            continue
        if parts[1] == "list":
            return None
        prop_types.append(parts[1].lower())
    if len(prop_types) != 1:
        return None
    return prop_types[0]


def _read_ply_element_values(path: Path, element_name: str) -> list[float] | None:
    format_line, elements, data_start = _parse_ply_header(path)
    if not format_line.startswith("format binary_"):
        return None
    if "binary_little_endian" in format_line:
        endian = "<"
    elif "binary_big_endian" in format_line:
        endian = ">"
    else:
        return None

    offset = data_start
    for name, count, props in elements:
        record_size = _element_fixed_record_size(props)
        if name == element_name:
            prop_type = _element_single_property_type(props)
            if prop_type is None:
                raise RuntimeError(f"PLY element '{element_name}' is not a single scalar property.")
            struct_char = _PLY_SCALAR_TYPE_STRUCT.get(prop_type)
            if struct_char is None:
                raise RuntimeError(f"Unsupported PLY scalar type: {prop_type}")
            if count <= 0:
                return []
            byte_count = count * _PLY_SCALAR_TYPE_SIZES[prop_type]
            with path.open("rb") as f:
                f.seek(offset)
                data = f.read(byte_count)
            if len(data) != byte_count:
                raise RuntimeError(f"Unexpected EOF while reading PLY element '{element_name}'.")
            fmt = endian + struct_char
            return [val[0] for val in struct.iter_unpack(fmt, data)]
        offset += count * record_size
    return None


def _read_ply_camera_meta(
    path: Path,
) -> tuple[float, tuple[int, int], float] | None:
    if path.suffix.lower() != ".ply":
        return None

    intrinsic = _read_ply_element_values(path, "intrinsic")
    if not intrinsic:
        return None

    image_size = _read_ply_element_values(path, "image_size")
    if image_size and len(image_size) >= 2:
        width = float(image_size[0])
        height = float(image_size[1])
    elif len(intrinsic) == 4:
        width = float(intrinsic[2])
        height = float(intrinsic[3])
    else:
        return None

    if len(intrinsic) >= 5:
        f_px = float(intrinsic[4])
    elif len(intrinsic) >= 2:
        f_px = float(intrinsic[1])
    else:
        f_px = float(intrinsic[0])

    depth_focus = 2.0
    disparity = _read_ply_element_values(path, "disparity")
    if disparity and len(disparity) >= 2:
        q90 = float(disparity[1])
        if q90 > 1e-6:
            depth_focus = max(depth_focus, 1.0 / q90)

    return f_px, (int(width), int(height)), depth_focus


def _compute_gsplat_camera_settings(
    path: Path,
) -> tuple[dict[str, object], float | None, float] | None:
    meta = _read_ply_camera_meta(path)
    if meta is None:
        return None

    f_px, resolution_px, depth_focus = meta
    height = float(resolution_px[1])

    fov = None
    if f_px > 0 and height > 0:
        fov = 2.0 * math.degrees(math.atan((height * 0.5) / f_px))

    camera_settings: dict[str, object] = {
        "camera": {
            "position": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, float(depth_focus)],
        }
    }
    if fov is not None:
        camera_settings["camera"]["fov"] = float(fov)

    return camera_settings, fov, depth_focus


def _interpolated_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values.sort()
    if len(values) == 1:
        return float(values[0])
    q = min(max(q, 0.0), 1.0)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    lower = values[lo]
    upper = values[hi]
    return float(lower + (upper - lower) * (pos - lo))


def _find_vertex_property_offset(
    prop_lines: list[str], name: str
) -> tuple[int, str] | None:
    offset = 0
    for line in prop_lines:
        parts = line.split()
        if len(parts) < 3 or parts[0] != "property":
            continue
        if parts[1] == "list":
            raise RuntimeError(
                "PLY contains variable-length list properties; cannot parse vertex depth."
            )
        prop_type = parts[1].lower()
        prop_name = parts[2].lower()
        size = _PLY_SCALAR_TYPE_SIZES.get(prop_type)
        if size is None:
            raise RuntimeError(f"Unsupported PLY scalar type: {prop_type}")
        if prop_name == name:
            return offset, prop_type
        offset += size
    return None


def _compute_depth_quantile_from_ply(path: Path, quantile: float) -> float | None:
    format_line, elements, data_start = _parse_ply_header(path)
    if not format_line.startswith("format binary_"):
        return None
    if "binary_little_endian" in format_line:
        endian = "<"
    elif "binary_big_endian" in format_line:
        endian = ">"
    else:
        return None

    offset = data_start
    vertex_offset = None
    vertex_count = 0
    vertex_record_size = 0
    z_offset = None
    z_type = None
    for name, count, props in elements:
        record_size = _element_fixed_record_size(props)
        if name == "vertex":
            vertex_offset = offset
            vertex_count = count
            vertex_record_size = record_size
            z_info = _find_vertex_property_offset(props, "z")
            if z_info is None:
                return None
            z_offset, z_type = z_info
            break
        offset += count * record_size

    if vertex_offset is None or vertex_count <= 0 or z_offset is None or z_type is None:
        return None

    struct_char = _PLY_SCALAR_TYPE_STRUCT.get(z_type)
    if struct_char is None:
        raise RuntimeError(f"Unsupported PLY scalar type: {z_type}")

    z_values: list[float] = []
    z_struct = struct.Struct(endian + struct_char)
    with path.open("rb") as f:
        f.seek(vertex_offset)
        remaining = vertex_count
        batch_records = 16_384
        while remaining > 0:
            batch = min(batch_records, remaining)
            data = f.read(batch * vertex_record_size)
            if len(data) != batch * vertex_record_size:
                raise RuntimeError("Unexpected EOF while reading PLY vertices.")
            base = 0
            for _ in range(batch):
                z_val = z_struct.unpack_from(data, base + z_offset)[0]
                if z_val > 0:
                    z_values.append(float(z_val))
                base += vertex_record_size
            remaining -= batch

    return _interpolated_quantile(z_values, quantile)


def _compute_max_offset_xyz(
    *,
    min_depth: float,
    resolution_px: tuple[int, int],
    f_px: float,
    max_disparity: float,
    max_zoom: float,
) -> list[float] | None:
    width_px, height_px = resolution_px
    if min_depth <= 0 or height_px <= 0 or f_px <= 0:
        return None
    aspect_ratio = width_px / height_px if height_px else 1.0
    reference_aspect = 21 / 9
    horizontal_scale = min(1.0, aspect_ratio / reference_aspect)
    diagonal = math.sqrt((width_px / f_px) ** 2 + (height_px / f_px) ** 2)
    max_lateral_offset = max_disparity * diagonal * min_depth * MOTION_AMPLITUDE_SCALE
    max_medial_offset = max_zoom * min_depth * MOTION_AMPLITUDE_SCALE
    return [
        max_lateral_offset * horizontal_scale,
        max_lateral_offset,
        max_medial_offset,
    ]


def _format_motion_label(traj_type: str, index: int) -> str:
    return f"{traj_type.replace('_', ' ').title()} {index + 1}"


def _compute_gsplat_motion_presets(path: Path) -> dict[str, object] | None:
    meta = _read_ply_camera_meta(path)
    if meta is None:
        return None

    f_px, resolution_px, _depth_focus = meta
    min_depth = _compute_depth_quantile_from_ply(path, 0.001)
    if min_depth is None:
        return None

    base_max_disparity = 0.08
    base_max_zoom = 0.15
    base_distance = 0.0
    base_num_steps = 60
    preset_count = 5
    variant_types = [
        "rotate_forward",
        "rotate",
        "swipe",
        "shake",
        "rotate_forward",
    ]

    presets: list[dict[str, object]] = []
    for variant_index in range(preset_count):
        rng = random.Random(variant_index)
        disparity_scale = rng.uniform(0.6, 1.6)
        zoom_scale = rng.uniform(0.6, 1.6)
        distance_offset = rng.uniform(-0.2, 0.2)
        repeats = rng.choice([1, 2, 3])
        lookat_mode = rng.choice(["point", "ahead"])
        traj_type = variant_types[variant_index % len(variant_types)]
        max_offset = _compute_max_offset_xyz(
            min_depth=min_depth,
            resolution_px=resolution_px,
            f_px=f_px,
            max_disparity=base_max_disparity * disparity_scale,
            max_zoom=base_max_zoom * zoom_scale,
        )
        if max_offset is None:
            continue
        presets.append(
            {
                "id": f"v{variant_index:02d}",
                "label": _format_motion_label(traj_type, variant_index),
                "type": traj_type,
                "lookat": lookat_mode,
                "maxOffset": [round(val, 6) for val in max_offset],
                "distance": round(base_distance + distance_offset, 6),
                "numSteps": base_num_steps,
                "numRepeats": repeats,
            }
        )

    if not presets:
        return None

    return {"version": 1, "fps": MOTION_PRESET_FPS, "presets": presets}


def _merge_viewer_settings(
    base: dict[str, object], overrides: dict[str, object]
) -> dict[str, object]:
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = {**existing, **value}
        else:
            merged[key] = value
    return merged


def _load_viewer_settings(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Viewer settings JSON must be an object.")
    return data


def _needs_vertex_only_ply(path: Path) -> bool:
    if path.suffix.lower() != ".ply":
        return False
    format_line, elements, _data_start = _parse_ply_header(path)
    if not format_line.startswith("format binary_"):
        # splat-transform supports ASCII PLY too, but SHARP emits binary; keep this simple.
        return False
    return not (len(elements) == 1 and elements[0][0] == "vertex")


def _write_vertex_only_ply(src: Path, dst: Path) -> None:
    format_line, elements, data_start = _parse_ply_header(src)
    if not format_line.startswith("format binary_"):
        raise RuntimeError(f"Unsupported PLY format for stripping: {format_line}")

    element_names = [name for name, _count, _props in elements]
    if "vertex" not in element_names:
        raise RuntimeError("PLY does not contain a 'vertex' element.")
    vertex_index = element_names.index("vertex")

    sizes = []
    for name, count, props in elements:
        record_size = _element_fixed_record_size(props)
        sizes.append((name, count, record_size, count * record_size))

    vertex_name, vertex_count, vertex_props = elements[vertex_index]
    vertex_total_size = sizes[vertex_index][3]
    vertex_offset = data_start + sum(total for _n, _c, _r, total in sizes[:vertex_index])

    header_lines = [
        "ply",
        format_line,
        "comment generated by sharp-splat-viewer-gui (vertex-only copy for splat-transform)",
        f"element {vertex_name} {vertex_count}",
        *vertex_props,
        "end_header",
    ]
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f_in, dst.open("wb") as f_out:
        f_out.write(header)
        f_in.seek(vertex_offset)
        remaining = vertex_total_size
        while remaining:
            chunk = f_in.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("Unexpected EOF while copying vertex data.")
            f_out.write(chunk)
            remaining -= len(chunk)


def _find_node_and_cli() -> tuple[Path, Path, Path]:
    """Return (node_exe, cli_mjs, vendor_root)."""
    candidate_roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate_roots.append(Path(meipass))

    exe_dir = Path(sys.executable).resolve().parent
    candidate_roots += [exe_dir, exe_dir / "_internal"]

    for root in dict.fromkeys(candidate_roots):
        node_exe = root / "node" / "node.exe"
        vendor_root = root / "splat-transform"
        cli_mjs = (
            vendor_root
            / "node_modules"
            / "@playcanvas"
            / "splat-transform"
            / "bin"
            / "cli.mjs"
        )
        if node_exe.is_file() and cli_mjs.is_file():
            return node_exe, cli_mjs, vendor_root

    # Fall back to system node + global install if present.
    node_on_path = shutil.which("node.exe") or shutil.which("node")
    splat_transform_on_path = shutil.which("splat-transform.cmd") or shutil.which("splat-transform")
    if node_on_path and splat_transform_on_path:
        # Best-effort: assume `splat-transform` is runnable directly.
        return Path(node_on_path), Path(splat_transform_on_path), Path.cwd()

    raise RuntimeError(
        "Could not find bundled Node/splat-transform. Rebuild the app, or install Node and "
        "`npm install -g @playcanvas/splat-transform` and try again."
    )


def _build_command(cfg: RunConfig) -> tuple[list[str], Path]:
    node_exe, cli, vendor_root = _find_node_and_cli()

    # If cli is actually `splat-transform.cmd`, we can invoke it directly without node.
    if cli.name.lower().endswith(".cmd"):
        base = [str(cli)]
    else:
        base = [str(node_exe), str(cli)]

    args: list[str] = []
    if cfg.overwrite:
        args += ["-w"]
    if cfg.viewer_settings is not None:
        args += ["-E", str(cfg.viewer_settings)]
    if cfg.unbundled:
        args += ["-U"]
    if cfg.gpu:
        args += ["-g", cfg.gpu]
    if cfg.iterations is not None:
        args += ["-i", str(int(cfg.iterations))]

    args += [str(cfg.input_path), str(cfg.output_html)]

    if cfg.extra_args.strip():
        args += shlex.split(cfg.extra_args, posix=os.name != "nt")

    return base + args, vendor_root


def generate_viewer(cfg: RunConfig, *, log_line: Callable[[str], None] | None = None) -> None:
    """Generate a self-contained HTML viewer from a SHARP Gaussians PLY."""

    def emit(msg: str) -> None:
        if log_line is not None:
            log_line(msg)

    started = time.time()
    effective_cfg = cfg
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if cfg.match_gsplat_camera:
            try:
                camera_result = _compute_gsplat_camera_settings(cfg.input_path)
            except Exception as exc:
                emit(f"Could not derive gsplat camera settings: {exc}")
            else:
                if camera_result is None:
                    emit("Could not derive gsplat camera settings; using defaults.")
                else:
                    camera_settings, fov, depth_focus = camera_result
                    if fov is None:
                        fov_label = "default fov"
                    else:
                        fov_label = f"fov={fov:.2f}deg"
                    emit(f"Using gsplat camera settings ({fov_label}, focus={depth_focus:.2f}m).")
                    merged_settings = camera_settings
                    if cfg.viewer_settings is not None:
                        existing = _load_viewer_settings(cfg.viewer_settings)
                        merged_settings = _merge_viewer_settings(existing, camera_settings)
                    if temp_dir is None:
                        temp_dir = tempfile.TemporaryDirectory(prefix="sharp-splat-transform-")
                    settings_path = Path(temp_dir.name) / "viewer.settings.json"
                    settings_path.write_text(
                        json.dumps(merged_settings, indent=2),
                        encoding="utf-8",
                    )
                    effective_cfg = dataclasses.replace(
                        effective_cfg, viewer_settings=settings_path
                    )

        motion_presets = None
        try:
            motion_presets = _compute_gsplat_motion_presets(cfg.input_path)
            if motion_presets is not None:
                emit("Embedding gsplat motion presets in viewer.")
        except Exception as exc:
            emit(f"Could not derive gsplat motion presets: {exc}")

        if _needs_vertex_only_ply(effective_cfg.input_path):
            emit(
                "Input PLY contains extra metadata elements; generating a vertex-only copy for "
                "splat-transform..."
            )
            if temp_dir is None:
                temp_dir = tempfile.TemporaryDirectory(prefix="sharp-splat-transform-")
            stripped = Path(temp_dir.name) / f"{effective_cfg.input_path.stem}.vertex-only.ply"
            _write_vertex_only_ply(effective_cfg.input_path, stripped)
            effective_cfg = dataclasses.replace(effective_cfg, input_path=stripped)

        cmd, cwd = _build_command(effective_cfg)
        emit("Running: " + " ".join(cmd))

        effective_cfg.output_html.parent.mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            emit(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"splat-transform exited with code {rc}")

        patch_bits = [
            f"AA={'on' if effective_cfg.enable_aa else 'off'}",
            f"MSAA={'on' if effective_cfg.enable_msaa else 'off'}",
        ]
        if motion_presets is not None:
            patch_bits.append("gsplat motion presets")
        emit("Applying viewer defaults (" + ", ".join(patch_bits) + ")...")
        _apply_viewer_customizations(
            effective_cfg.output_html,
            unbundled=effective_cfg.unbundled,
            enable_aa=effective_cfg.enable_aa,
            enable_msaa=effective_cfg.enable_msaa,
            motion_presets=motion_presets,
        )

        emit(f"Done in {time.time() - started:.1f}s")
    finally:
        try:
            if temp_dir is not None:
                temp_dir.cleanup()
        except Exception:
            pass


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("SHARP -> PlayCanvas HTML Viewer (splat-transform)")
    root.geometry("980x700")

    output_q: queue.Queue[str] = queue.Queue()
    is_running = tk.BooleanVar(value=False)

    def log_line(s: str) -> None:
        output_q.put(s + ("\n" if not s.endswith("\n") else ""))

    def drain_output() -> None:
        try:
            while True:
                chunk = output_q.get_nowait()
                log_text.configure(state="normal")
                log_text.insert("end", chunk)
                log_text.see("end")
                log_text.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(100, drain_output)

    def set_running(running: bool) -> None:
        is_running.set(running)
        run_btn.configure(state=("disabled" if running else "normal"))
        copy_btn.configure(state=("disabled" if running else "normal"))

    def browse_input() -> None:
        initial_dir = "C:/Users/phill/Dropbox/Splats/outputs"
        initialdir_arg = {"initialdir": initial_dir} if Path(initial_dir).exists() else {}
        p = filedialog.askopenfilename(
            title="Select a .ply file",
            filetypes=[("PLY", "*.ply"), ("All", "*.*")],
            **initialdir_arg,
        )
        if p:
            input_var.set(p)
            if not output_var.get().strip():
                out_dir = Path(output_folder_var.get().strip() or "C:/Users/phill/Dropbox/Splats/outputs")
                output_var.set(str(out_dir / (Path(p).stem + ".html")))

    def browse_output_html() -> None:
        default_out = output_var.get().strip()
        initial_dir = Path(default_out).parent if default_out else Path(output_folder_var.get())
        initialdir_arg = {"initialdir": str(initial_dir)} if initial_dir.exists() else {}
        p = filedialog.asksaveasfilename(
            title="Save HTML viewer as...",
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("All", "*.*")],
            **initialdir_arg,
        )
        if p:
            output_var.set(p)

    def browse_viewer_settings() -> None:
        p = filedialog.askopenfilename(
            title="Select viewer settings JSON (optional)",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if p:
            settings_var.set(p)

    def open_output_html() -> None:
        p = output_var.get().strip()
        if not p:
            return
        try:
            os.startfile(p)  # type: ignore[attr-defined]
        except Exception as exc:
            log_line(f"Could not open output: {exc}")

    def make_config() -> RunConfig | None:
        in_path = input_var.get().strip()
        out_html = output_var.get().strip()
        if not in_path or not out_html:
            log_line("Please select both input and output paths.")
            return None

        settings = settings_var.get().strip()
        settings_path = Path(settings) if settings else None
        gpu = gpu_var.get().strip()
        gpu_value = None
        if gpu and gpu.lower() != "default":
            gpu_value = gpu

        iters = iterations_var.get().strip()
        iterations = int(iters) if iters else None

        return RunConfig(
            input_path=Path(in_path),
            output_html=Path(out_html),
            viewer_settings=settings_path,
            unbundled=bool(unbundled_var.get()),
            overwrite=bool(overwrite_var.get()),
            enable_aa=bool(enable_aa_var.get()),
            enable_msaa=bool(enable_msaa_var.get()),
            match_gsplat_camera=bool(match_gsplat_camera_var.get()),
            gpu=gpu_value,
            iterations=iterations,
            extra_args=extra_args_var.get(),
        )

    def copy_command() -> None:
        cfg = make_config()
        if cfg is None:
            return
        try:
            cmd, _cwd = _build_command(cfg)
        except Exception as exc:
            log_line(f"ERROR: {exc}")
            return
        root.clipboard_clear()
        root.clipboard_append(" ".join(cmd))
        log_line("Copied command to clipboard.")

    def show_help() -> None:
        try:
            node_exe, cli, vendor_root = _find_node_and_cli()
            if cli.name.lower().endswith(".cmd"):
                cmd = [str(cli), "--help"]
                cwd = None
            else:
                cmd = [str(node_exe), str(cli), "--help"]
                cwd = vendor_root
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            log_line("\n" + (proc.stdout or proc.stderr or "(no output)") + "\n")
        except Exception as exc:
            log_line(f"ERROR: {exc}")

    def run_in_thread(cfg: RunConfig) -> None:
        set_running(True)
        try:
            generate_viewer(cfg, log_line=log_line)
        except Exception as exc:
            log_line(f"ERROR: {exc}")
        finally:
            set_running(False)

    def on_run() -> None:
        if is_running.get():
            return
        cfg = make_config()
        if cfg is None:
            return
        threading.Thread(target=run_in_thread, args=(cfg,), daemon=True).start()

    def on_clear_log() -> None:
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")

    # Vars
    input_var = tk.StringVar(value="")
    output_folder_var = tk.StringVar(value="C:/Users/phill/Dropbox/Splats/outputs")
    output_var = tk.StringVar(value="")
    settings_var = tk.StringVar(value="")
    unbundled_var = tk.IntVar(value=0)
    overwrite_var = tk.IntVar(value=1)
    match_gsplat_camera_var = tk.IntVar(value=1)
    enable_aa_var = tk.IntVar(value=1)
    enable_msaa_var = tk.IntVar(value=1)
    gpu_var = tk.StringVar(value="default")
    iterations_var = tk.StringVar(value="18")
    extra_args_var = tk.StringVar(value="")

    # Layout
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    row = 0
    ttk.Label(frm, text="Input (.ply)").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=input_var, width=86).grid(row=row, column=1, sticky="we")
    ttk.Button(frm, text="Browse...", command=browse_input).grid(row=row, column=2, padx=(8, 0))

    row += 1
    ttk.Label(frm, text="Output (.html)").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frm, textvariable=output_var, width=86).grid(
        row=row, column=1, sticky="we", pady=(8, 0)
    )
    ttk.Button(frm, text="Save As...", command=browse_output_html).grid(
        row=row, column=2, padx=(8, 0), pady=(8, 0)
    )

    row += 1
    ttk.Label(frm, text="Viewer settings (.json)").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frm, textvariable=settings_var, width=86).grid(
        row=row, column=1, sticky="we", pady=(8, 0)
    )
    ttk.Button(frm, text="Browse...", command=browse_viewer_settings).grid(
        row=row, column=2, padx=(8, 0), pady=(8, 0)
    )

    row += 1
    opts = ttk.Frame(frm)
    opts.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
    ttk.Checkbutton(opts, text="Overwrite output", variable=overwrite_var).pack(side="left")
    ttk.Checkbutton(opts, text="Unbundled output (-U)", variable=unbundled_var).pack(
        side="left", padx=(12, 0)
    )
    ttk.Checkbutton(opts, text="Match gsplat camera", variable=match_gsplat_camera_var).pack(
        side="left", padx=(12, 0)
    )

    row += 1
    quality_opts = ttk.Frame(frm)
    quality_opts.grid(row=row, column=0, columnspan=3, sticky="we", pady=(6, 0))
    ttk.Checkbutton(quality_opts, text="Enable splat AA", variable=enable_aa_var).pack(side="left")
    ttk.Checkbutton(quality_opts, text="Enable MSAA", variable=enable_msaa_var).pack(
        side="left", padx=(12, 0)
    )
    ttk.Label(quality_opts, text="GPU").pack(side="left", padx=(12, 0))
    ttk.Entry(quality_opts, textvariable=gpu_var, width=10).pack(side="left")
    ttk.Label(quality_opts, text="Iterations").pack(side="left", padx=(12, 0))
    ttk.Entry(quality_opts, textvariable=iterations_var, width=8).pack(side="left")

    row += 1
    ttk.Label(frm, text="Extra args").grid(row=row, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(frm, textvariable=extra_args_var, width=86).grid(
        row=row, column=1, columnspan=2, sticky="we", pady=(10, 0)
    )

    row += 1
    btns = ttk.Frame(frm)
    btns.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 10))
    run_btn = ttk.Button(btns, text="Generate Viewer", command=on_run)
    run_btn.pack(side="left")
    ttk.Button(btns, text="Open Output", command=open_output_html).pack(side="left", padx=(8, 0))
    copy_btn = ttk.Button(btns, text="Copy Command", command=copy_command)
    copy_btn.pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="Show splat-transform Help", command=show_help).pack(
        side="left", padx=(8, 0)
    )
    ttk.Button(btns, text="Clear Log", command=on_clear_log).pack(side="left", padx=(8, 0))

    row += 1
    ttk.Label(frm, text="Output").grid(row=row, column=0, sticky="w")
    row += 1
    log_text = tk.Text(frm, height=24, wrap="word", state="disabled")
    log_text.grid(row=row, column=0, columnspan=3, sticky="nsew")
    scrollbar = ttk.Scrollbar(frm, command=log_text.yview)
    scrollbar.grid(row=row, column=3, sticky="ns")
    log_text.configure(yscrollcommand=scrollbar.set)

    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(row, weight=1)

    drain_output()
    log_line("Tip: Defaults to a bundled single-page HTML viewer. Use -U for unbundled output.")
    root.mainloop()
