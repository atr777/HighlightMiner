"""Crop positioning page.

Where the 9:16 window sits decides what every clip from a channel looks like.
This shows real frames from across the VOD with the window drawn on them, and
the resulting crop beside each, so the choice is judged rather than guessed.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import streamlit as st

from .crop_tool import (
    DEFAULT_SAMPLE_COUNT,
    CropToolError,
    annotate_frame,
    crop_preview,
    rect_for_aspect,
    sample_frames,
)
from .media import probe_media
from .render import Rect
from .security import validate_local_video
from .settings_store import load_app_settings, save_app_settings
from .storage import list_analyses
from .ui_common import _VIDEO_FILTER, path_picker

_STATE_PREFIX = "crop_tool_"
_FRAMES_KEY = f"{_STATE_PREFIX}frames"
_SOURCE_KEY = f"{_STATE_PREFIX}source"
_DURATION_KEY = f"{_STATE_PREFIX}duration"

_X_KEY = f"{_STATE_PREFIX}x"
_Y_KEY = f"{_STATE_PREFIX}y"
_H_KEY = f"{_STATE_PREFIX}h"
_COUNT_KEY = f"{_STATE_PREFIX}count"


def _work_dir() -> Path:
    directory = Path(tempfile.gettempdir()) / "highlightminer-crop-tool"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _recent_sources(db_path: Path, limit: int = 12) -> list[str]:
    seen: list[str] = []
    try:
        for row in list_analyses(db_path, limit=limit * 3):
            path = str(row.get("video_path") or "")
            if path and path not in seen and Path(path).exists():
                seen.append(path)
            if len(seen) >= limit:
                break
    except Exception:
        return []
    return seen


def _seed_from_settings(settings) -> None:
    rect = settings.gameplay_rect
    if _X_KEY in st.session_state:
        return
    if rect:
        st.session_state[_X_KEY] = float(rect.get("x", 0.341797))
        st.session_state[_Y_KEY] = float(rect.get("y", 0.0))
        st.session_state[_H_KEY] = float(rect.get("h", 1.0))
    else:
        centred = rect_for_aspect(x=0.0, height=1.0)
        st.session_state[_X_KEY] = round((1.0 - centred.w) / 2, 6)
        st.session_state[_Y_KEY] = 0.0
        st.session_state[_H_KEY] = 1.0
    st.session_state.setdefault(_COUNT_KEY, DEFAULT_SAMPLE_COUNT)


def _current_rect() -> Rect:
    return rect_for_aspect(
        x=float(st.session_state[_X_KEY]),
        y=float(st.session_state[_Y_KEY]),
        height=float(st.session_state[_H_KEY]),
    )


def _centre_horizontally() -> None:
    rect = _current_rect()
    st.session_state[_X_KEY] = round((1.0 - rect.w) / 2, 6)


def _reset() -> None:
    st.session_state[_Y_KEY] = 0.0
    st.session_state[_H_KEY] = 1.0
    _centre_horizontally()


def render_crop_page(db_path: Path) -> None:
    settings = load_app_settings(db_path)
    _seed_from_settings(settings)

    st.header("🎯 Crop position", anchor=False)
    st.caption(
        "A 9:16 crop of a 16:9 frame keeps about a third of the width. Sample frames "
        "from across the VOD, place the window, and save it as the crop region used "
        "for every export."
    )

    if settings.render_layout != "crop":
        st.info(
            f"The active layout is **{settings.render_layout}**, so this region will not be "
            "used until you switch the layout to **crop** in Settings."
        )

    recent = _recent_sources(db_path)
    if recent:
        chosen = st.selectbox(
            "Recent source",
            ["(pick a file below)"] + recent,
            key=f"{_STATE_PREFIX}recent",
        )
        if chosen != "(pick a file below)":
            st.session_state.setdefault("crop_video_input", chosen)

    video_path = path_picker(
        "VOD",
        "crop_video_input",
        placeholder=r"H:\vod-scrape\vods\stream.mp4",
        file_filter=_VIDEO_FILTER,
    )

    st.slider("Frames to sample", 2, 12, key=_COUNT_KEY)

    if st.button("Sample frames", disabled=not video_path, type="primary"):
        try:
            source = validate_local_video(video_path)
            duration = float(probe_media(source)["duration"])
            with st.spinner("Extracting frames…"):
                frames = sample_frames(
                    source, duration, _work_dir(), count=int(st.session_state[_COUNT_KEY])
                )
        except (CropToolError, Exception) as exc:  # noqa: BLE001 - surfaced to the user
            st.error(f"Could not sample this VOD: {exc}")
        else:
            st.session_state[_FRAMES_KEY] = [(f.time, str(f.path)) for f in frames]
            st.session_state[_SOURCE_KEY] = str(source)
            st.session_state[_DURATION_KEY] = duration
            st.rerun()

    frames = st.session_state.get(_FRAMES_KEY)
    if not frames:
        st.caption("Pick a VOD and sample some frames to begin.")
        return

    duration = float(st.session_state.get(_DURATION_KEY) or 0.0)
    st.caption(
        f"{len(frames)} frames from `{Path(st.session_state[_SOURCE_KEY]).name}` "
        f"({duration / 3600:.2f} hours)"
    )

    st.subheader("Window", anchor=False)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider("Horizontal position", 0.0, 1.0, step=0.005, key=_X_KEY)
        st.slider("Vertical position", 0.0, 1.0, step=0.005, key=_Y_KEY)
        st.slider(
            "Height kept", 0.3, 1.0, step=0.01, key=_H_KEY,
            help="Below 1.0 crops off the top and bottom too, then the width narrows to match 9:16.",
        )
    with c2:
        st.button("Centre", on_click=_centre_horizontally, width="stretch")
        st.button("Reset", on_click=_reset, width="stretch")

    rect = _current_rect()
    st.caption(
        f"Region x={rect.x:.3f} y={rect.y:.3f} w={rect.w:.3f} h={rect.h:.3f} · "
        f"keeps {rect.w * 100:.0f}% of the width"
    )

    st.subheader("Preview", anchor=False)
    for when, frame_path in frames:
        left, right = st.columns([2, 1])
        stem = Path(frame_path).stem
        try:
            annotated = annotate_frame(frame_path, rect, _work_dir() / f"{stem}_marked.png")
            cropped = crop_preview(frame_path, rect, _work_dir() / f"{stem}_crop.png")
        except Exception as exc:  # noqa: BLE001 - a bad frame should not break the page
            st.warning(f"Could not render the frame at {when:.0f}s: {exc}")
            continue
        left.image(str(annotated), caption=f"{when / 60:.1f} min — kept region")
        right.image(str(cropped), caption="Result")

    st.divider()
    if st.button("Save as the crop region", type="primary"):
        save_app_settings(
            dataclasses.replace(settings, gameplay_rect=rect.as_dict()), db_path
        )
        st.success(
            f"Saved. Exports using the crop layout will use x={rect.x:.3f}, w={rect.w:.3f}."
        )
