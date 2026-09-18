from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st

from .config import Settings, _STANDARD_WHISPER_MODELS
from .runtime import app_root
from .settings_presets import (
    TIMING_PRESETS,
    apply_timing_preset,
    detect_timing_preset,
    WEIGHT_PRESETS,
    detect_weight_preset,
    normalize_weights,
)
from .settings_store import export_app_settings, import_app_settings, load_app_settings, reset_app_settings, save_app_settings
from .ui_common import _JSON_FILTER, choose_save_file, path_picker
from .ui_diagnostics import render_diagnostics_settings
from .ui_model_access import render_model_access_settings

_EDITOR_KEYS = {
    "model": "cfg_model",
    "custom_model": "cfg_custom_model",
    "device": "cfg_device",
    "compute": "cfg_compute",
    "language": "cfg_language",
    "beam": "cfg_beam",
    "vad": "cfg_vad",
    "audio_window": "cfg_audio_window",
    "audio_hop": "cfg_audio_hop",
    "pre_roll": "cfg_pre_roll",
    "post_roll": "cfg_post_roll",
    "merge_gap": "cfg_merge_gap",
    "max_clip": "cfg_max_clip",
    "min_score": "cfg_min_score",
    "max_candidates": "cfg_max_candidates",
    "audio_weight": "cfg_audio_weight",
    "transcript_weight": "cfg_transcript_weight",
    "chat_weight": "cfg_chat_weight",
    "reactions": "cfg_reactions",
    "preset": "cfg_preset",
    "timing_preset": "cfg_timing_preset",
    "short_form": "cfg_short_form",
    "hook_lead": "cfg_hook_lead",
    "min_clip": "cfg_min_clip",
    "speech_snap": "cfg_speech_snap",
    "render_layout": "cfg_render_layout",
    "burn_captions": "cfg_burn_captions",
    "caption_size": "cfg_caption_size",
    "caption_upper": "cfg_caption_upper",
    "webcam_fraction": "cfg_webcam_fraction",
    "audio_only_penalty": "cfg_audio_only_penalty",
    "duplicate_containment": "cfg_duplicate_containment",
    "chat_quiet": "cfg_chat_quiet",
    "chat_active": "cfg_chat_active",
    "chat_burst": "cfg_chat_burst",
    "cpu_threads": "cfg_cpu_threads",
    "webcam_x": "cfg_webcam_x",
    "webcam_y": "cfg_webcam_y",
    "webcam_w": "cfg_webcam_w",
    "webcam_h": "cfg_webcam_h",
    "webcam_enabled": "cfg_webcam_enabled",
    "crop_enabled": "cfg_crop_enabled",
    "crop_x": "cfg_crop_x",
    "crop_y": "cfg_crop_y",
    "crop_w": "cfg_crop_w",
    "crop_h": "cfg_crop_h",
}

_PRIMARY_WHISPER_MODELS = ("large-v3", "turbo", "medium", "small")
_ADVANCED_WHISPER_MODEL = "Other / advanced…"


def _model_editor_values(settings: Settings) -> tuple[str, str]:
    if settings.whisper_model in _PRIMARY_WHISPER_MODELS:
        return settings.whisper_model, ""
    return _ADVANCED_WHISPER_MODEL, settings.whisper_model


def _editor_needs_seed(state: Mapping[str, Any]) -> bool:
    required_names = [name for name in _EDITOR_KEYS if name != "custom_model"]
    return any(_EDITOR_KEYS[name] not in state for name in required_names)


def _seed_editor(settings: Settings, *, force: bool = False) -> None:
    if not force and not _editor_needs_seed(st.session_state):
        return
    normalized = normalize_weights(settings.weights)
    model, custom_model = _model_editor_values(settings)
    values = {
        "model": model,
        "custom_model": custom_model,
        "device": settings.device,
        "compute": settings.compute_type,
        "language": settings.language or "",
        "beam": int(settings.beam_size),
        "vad": bool(settings.vad_filter),
        "audio_window": float(settings.audio_window_sec),
        "audio_hop": float(settings.audio_hop_sec),
        "pre_roll": float(settings.pre_roll_sec),
        "post_roll": float(settings.post_roll_sec),
        "merge_gap": float(settings.merge_gap_sec),
        "max_clip": float(settings.max_candidate_sec),
        "min_score": float(settings.min_candidate_score),
        "max_candidates": int(settings.max_candidates),
        "audio_weight": normalized["audio"],
        "transcript_weight": normalized["transcript"],
        "chat_weight": normalized["chat"],
        "reactions": "\n".join(settings.reaction_phrases),
        "preset": detect_weight_preset(normalized),
        "timing_preset": detect_timing_preset(settings),
        "short_form": bool(settings.short_form_mode),
        "hook_lead": float(settings.hook_lead_sec),
        "min_clip": float(settings.min_candidate_sec),
        "speech_snap": float(settings.speech_snap_sec),
        "render_layout": settings.render_layout,
        "burn_captions": bool(settings.burn_captions),
        "caption_size": int(settings.caption_font_size),
        "caption_upper": bool(settings.caption_uppercase),
        "webcam_fraction": float(settings.webcam_fraction),
        "audio_only_penalty": float(settings.audio_only_penalty),
        "duplicate_containment": float(settings.duplicate_containment),
        "chat_quiet": float(settings.chat_quiet_msgs_per_min),
        "chat_active": float(settings.chat_active_msgs_per_min),
        "chat_burst": float(settings.chat_min_burst_messages),
        "cpu_threads": int(settings.cpu_threads),
        "webcam_enabled": settings.webcam_rect is not None,
        "webcam_x": float((settings.webcam_rect or {}).get("x", 0.70)),
        "webcam_y": float((settings.webcam_rect or {}).get("y", 0.0)),
        "webcam_w": float((settings.webcam_rect or {}).get("w", 0.30)),
        "webcam_h": float((settings.webcam_rect or {}).get("h", 0.30)),
        "crop_enabled": settings.gameplay_rect is not None,
        "crop_x": float((settings.gameplay_rect or {}).get("x", 0.341797)),
        "crop_y": float((settings.gameplay_rect or {}).get("y", 0.0)),
        "crop_w": float((settings.gameplay_rect or {}).get("w", 0.316406)),
        "crop_h": float((settings.gameplay_rect or {}).get("h", 1.0)),
    }
    for name, value in values.items():
        st.session_state[_EDITOR_KEYS[name]] = value



def _editor_webcam_rect() -> dict[str, float] | None:
    """The webcam rectangle, or None when the layout does not use one."""
    if not st.session_state.get(_EDITOR_KEYS["webcam_enabled"]):
        return None
    return {
        "x": float(st.session_state[_EDITOR_KEYS["webcam_x"]]),
        "y": float(st.session_state[_EDITOR_KEYS["webcam_y"]]),
        "w": float(st.session_state[_EDITOR_KEYS["webcam_w"]]),
        "h": float(st.session_state[_EDITOR_KEYS["webcam_h"]]),
    }


def _editor_gameplay_rect() -> dict[str, float] | None:
    """The crop region, or None to use the centred 9:16 slice."""
    if not st.session_state.get(_EDITOR_KEYS["crop_enabled"]):
        return None
    return {
        "x": float(st.session_state[_EDITOR_KEYS["crop_x"]]),
        "y": float(st.session_state[_EDITOR_KEYS["crop_y"]]),
        "w": float(st.session_state[_EDITOR_KEYS["crop_w"]]),
        "h": float(st.session_state[_EDITOR_KEYS["crop_h"]]),
    }


def _apply_timing_preset_to_editor() -> None:
    """Copy a chosen timing preset into the editor fields."""
    name = st.session_state.get(_EDITOR_KEYS["timing_preset"], "Review")
    preset = TIMING_PRESETS.get(name)
    if not preset:
        return
    mapping = {
        "short_form_mode": "short_form",
        "pre_roll_sec": "pre_roll",
        "post_roll_sec": "post_roll",
        "max_candidate_sec": "max_clip",
        "min_candidate_sec": "min_clip",
        "hook_lead_sec": "hook_lead",
        "min_candidate_score": "min_score",
        "max_candidates": "max_candidates",
        "audio_only_penalty": "audio_only_penalty",
        "duplicate_containment": "duplicate_containment",
    }
    for field, editor_name in mapping.items():
        if field in preset:
            st.session_state[_EDITOR_KEYS[editor_name]] = preset[field]


def _request_reload(message: str) -> None:
    st.session_state["settings_editor_reload"] = True
    st.session_state["settings_notice"] = message


def _editor_weights() -> dict[str, float]:
    return {
        "audio": float(st.session_state[_EDITOR_KEYS["audio_weight"]]),
        "transcript": float(st.session_state[_EDITOR_KEYS["transcript_weight"]]),
        "chat": float(st.session_state[_EDITOR_KEYS["chat_weight"]]),
    }


def _apply_selected_preset() -> None:
    """Fill the weight editor when the preset selector changes."""
    preset = st.session_state.get(_EDITOR_KEYS["preset"], "Balanced")
    if preset not in WEIGHT_PRESETS:
        return
    values = WEIGHT_PRESETS[preset]
    st.session_state[_EDITOR_KEYS["audio_weight"]] = values["audio"]
    st.session_state[_EDITOR_KEYS["transcript_weight"]] = values["transcript"]
    st.session_state[_EDITOR_KEYS["chat_weight"]] = values["chat"]


def _sync_preset_to_weights() -> None:
    """Keep manual slider edits represented as Custom (or an exact preset)."""
    st.session_state[_EDITOR_KEYS["preset"]] = detect_weight_preset(_editor_weights())


def _browse_export() -> None:
    try:
        current = st.session_state.get("settings_export_path", str(app_root() / "HighlightMiner-settings.json"))
        selected = choose_save_file("Export HighlightMiner settings", current)
        if selected:
            st.session_state["settings_export_path"] = selected
    except Exception as exc:
        st.session_state["native_dialog_error"] = str(exc)


def _build_settings() -> Settings:
    model_choice = str(st.session_state[_EDITOR_KEYS["model"]])
    advanced = model_choice == _ADVANCED_WHISPER_MODEL
    model = str(st.session_state.get(_EDITOR_KEYS["custom_model"], "")).strip() if advanced else model_choice
    phrases = [line.strip() for line in str(st.session_state[_EDITOR_KEYS["reactions"]]).splitlines() if line.strip()]
    return Settings(
        whisper_model=model,
        allow_custom_whisper_model=bool(advanced and model not in _STANDARD_WHISPER_MODELS),
        device=str(st.session_state[_EDITOR_KEYS["device"]]),
        compute_type=str(st.session_state[_EDITOR_KEYS["compute"]]),
        language=str(st.session_state[_EDITOR_KEYS["language"]]).strip() or None,
        beam_size=int(st.session_state[_EDITOR_KEYS["beam"]]),
        vad_filter=bool(st.session_state[_EDITOR_KEYS["vad"]]),
        audio_window_sec=float(st.session_state[_EDITOR_KEYS["audio_window"]]),
        audio_hop_sec=float(st.session_state[_EDITOR_KEYS["audio_hop"]]),
        pre_roll_sec=float(st.session_state[_EDITOR_KEYS["pre_roll"]]),
        post_roll_sec=float(st.session_state[_EDITOR_KEYS["post_roll"]]),
        merge_gap_sec=float(st.session_state[_EDITOR_KEYS["merge_gap"]]),
        max_candidate_sec=float(st.session_state[_EDITOR_KEYS["max_clip"]]),
        min_candidate_score=float(st.session_state[_EDITOR_KEYS["min_score"]]),
        max_candidates=int(st.session_state[_EDITOR_KEYS["max_candidates"]]),
        weights=_editor_weights(),
        reaction_phrases=phrases,
        short_form_mode=bool(st.session_state[_EDITOR_KEYS["short_form"]]),
        hook_lead_sec=float(st.session_state[_EDITOR_KEYS["hook_lead"]]),
        min_candidate_sec=float(st.session_state[_EDITOR_KEYS["min_clip"]]),
        speech_snap_sec=float(st.session_state[_EDITOR_KEYS["speech_snap"]]),
        render_layout=str(st.session_state[_EDITOR_KEYS["render_layout"]]),
        burn_captions=bool(st.session_state[_EDITOR_KEYS["burn_captions"]]),
        caption_font_size=int(st.session_state[_EDITOR_KEYS["caption_size"]]),
        caption_uppercase=bool(st.session_state[_EDITOR_KEYS["caption_upper"]]),
        webcam_fraction=float(st.session_state[_EDITOR_KEYS["webcam_fraction"]]),
        webcam_rect=_editor_webcam_rect(),
        gameplay_rect=_editor_gameplay_rect(),
        audio_only_penalty=float(st.session_state[_EDITOR_KEYS["audio_only_penalty"]]),
        duplicate_containment=float(st.session_state[_EDITOR_KEYS["duplicate_containment"]]),
        chat_quiet_msgs_per_min=float(st.session_state[_EDITOR_KEYS["chat_quiet"]]),
        chat_active_msgs_per_min=float(st.session_state[_EDITOR_KEYS["chat_active"]]),
        chat_min_burst_messages=float(st.session_state[_EDITOR_KEYS["chat_burst"]]),
        cpu_threads=int(st.session_state[_EDITOR_KEYS["cpu_threads"]]),
    )


def render_settings_page(db_path: Path) -> None:
    settings = load_app_settings(db_path)
    if st.session_state.pop("settings_editor_reload", False):
        _seed_editor(settings, force=True)
    else:
        _seed_editor(settings)

    st.header("⚙️ Settings", anchor=False)
    st.caption(
        "Analysis settings are stored in highlightminer.db and apply to future analyses and reruns. "
        "Existing runs keep their original snapshot; Model Access is saved separately as database-local security state."
    )
    notice = st.session_state.pop("settings_notice", None)
    if notice:
        st.success(notice)

    engine, detection, shortform, reactions, transfer = st.tabs(
        ["Analysis engine", "Detection & weights", "Short form", "Reaction phrases", "Import / Export"]
    )

    with engine:
        render_model_access_settings(db_path)
        st.divider()
        st.selectbox(
            "Whisper model",
            list(_PRIMARY_WHISPER_MODELS) + [_ADVANCED_WHISPER_MODEL],
            key=_EDITOR_KEYS["model"],
            help="large-v3 is HighlightMiner's default. Turbo is the faster general-purpose option. Use Other / advanced only when you deliberately want a different standard alias or custom Hugging Face model.",
        )
        if st.session_state[_EDITOR_KEYS["model"]] == _ADVANCED_WHISPER_MODEL:
            if _EDITOR_KEYS["custom_model"] not in st.session_state:
                st.session_state[_EDITOR_KEYS["custom_model"]] = ""
            st.text_input(
                "Other model name",
                key=_EDITOR_KEYS["custom_model"],
                placeholder="e.g. base.en or organization/model-name",
                help="Advanced opt-in. Standard faster-whisper aliases remain standard; arbitrary repositories are treated as custom models. A network download still requires the separate permission above.",
            )
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Device", ["auto", "cuda", "cpu"], key=_EDITOR_KEYS["device"])
            st.number_input("Beam size", min_value=1, max_value=20, step=1, key=_EDITOR_KEYS["beam"])
        with c2:
            st.selectbox("Compute type", ["auto", "float16", "float32", "int8", "int8_float16", "int8_float32", "bfloat16"], key=_EDITOR_KEYS["compute"])
            st.text_input("Language", key=_EDITOR_KEYS["language"], placeholder="blank = auto detect")
        st.checkbox("VAD filtering", key=_EDITOR_KEYS["vad"], help="Voice activity detection can skip long non-speech regions during transcription.")

    with detection:
        st.selectbox(
            "Signal weighting preset",
            list(WEIGHT_PRESETS) + ["Custom"],
            key=_EDITOR_KEYS["preset"],
            help="Selecting a preset fills the three weight sliders. Save analysis settings to make the editor values active.",
            on_change=_apply_selected_preset,
        )
        st.caption(
            "Choose a preset to fill the sliders. Manual edits show Custom unless the values match a built-in preset. "
            "Changes remain in the editor until you save analysis settings."
        )

        st.slider(
            "Audio weight",
            0.0,
            1.0,
            step=0.01,
            key=_EDITOR_KEYS["audio_weight"],
            on_change=_sync_preset_to_weights,
        )
        st.slider(
            "Transcript / reaction weight",
            0.0,
            1.0,
            step=0.01,
            key=_EDITOR_KEYS["transcript_weight"],
            on_change=_sync_preset_to_weights,
        )
        st.slider(
            "Chat weight",
            0.0,
            1.0,
            step=0.01,
            key=_EDITOR_KEYS["chat_weight"],
            on_change=_sync_preset_to_weights,
        )
        raw_weights = _editor_weights()
        if sum(raw_weights.values()) > 0:
            effective = normalize_weights(raw_weights)
            m1, m2, m3 = st.columns(3)
            m1.metric("Effective audio", f"{effective['audio']:.1%}")
            m2.metric("Effective transcript", f"{effective['transcript']:.1%}")
            m3.metric("Effective chat", f"{effective['chat']:.1%}")
            st.caption(
                f"Current weighting matches: **{detect_weight_preset(raw_weights)}**. "
                "If transcript or chat is unavailable during a run, that signal becomes 0% and the available signals are renormalized automatically."
            )
        else:
            st.error("At least one signal weight must be greater than zero.")

        c1, c2 = st.columns(2)
        with c1:
            st.slider("Minimum candidate score", 0.0, 1.0, step=0.01, key=_EDITOR_KEYS["min_score"])
            st.number_input("Maximum candidates", min_value=1, max_value=500, step=1, key=_EDITOR_KEYS["max_candidates"])
            st.number_input("Pre-roll (seconds)", min_value=0.0, max_value=600.0, step=1.0, key=_EDITOR_KEYS["pre_roll"])
            st.number_input("Post-roll (seconds)", min_value=0.0, max_value=600.0, step=1.0, key=_EDITOR_KEYS["post_roll"])
        with c2:
            st.number_input("Merge gap (seconds)", min_value=0.0, max_value=600.0, step=1.0, key=_EDITOR_KEYS["merge_gap"])
            st.number_input("Maximum clip length (seconds)", min_value=1.0, max_value=1800.0, step=1.0, key=_EDITOR_KEYS["max_clip"])
            st.number_input("Audio window (seconds)", min_value=0.1, max_value=10.0, step=0.1, key=_EDITOR_KEYS["audio_window"])
            st.number_input("Audio hop (seconds)", min_value=0.05, max_value=5.0, step=0.05, key=_EDITOR_KEYS["audio_hop"])


    with shortform:
        st.caption(
            "Short-form clips need the payoff in the first few seconds and a 9:16 frame. "
            "Timing presets only change clip shaping; signal weights and Whisper options are chosen separately."
        )

        st.selectbox(
            "Timing preset",
            list(TIMING_PRESETS) ,
            key=_EDITOR_KEYS["timing_preset"],
            on_change=_apply_timing_preset_to_editor,
            help="Review keeps generous context for judging a moment. Short-form trims to the hook.",
        )

        st.subheader("Clip shaping", anchor=False)
        st.toggle(
            "Reshape candidates for short form",
            key=_EDITOR_KEYS["short_form"],
            help=(
                "Trims so the peak lands near the front and snaps clip edges to speech "
                "boundaries, so a clip does not open or end mid-word."
            ),
        )
        s1, s2 = st.columns(2)
        with s1:
            st.number_input(
                "Hook lead (seconds)", min_value=0.0, max_value=60.0, step=0.5,
                key=_EDITOR_KEYS["hook_lead"],
                help="How far into the clip the peak moment should land.",
            )
            st.number_input(
                "Minimum clip length (seconds)", min_value=1.0, max_value=600.0, step=1.0,
                key=_EDITOR_KEYS["min_clip"],
            )
        with s2:
            st.number_input(
                "Speech snap tolerance (seconds)", min_value=0.0, max_value=30.0, step=0.1,
                key=_EDITOR_KEYS["speech_snap"],
            )
            st.slider(
                "Uncorroborated audio penalty", 0.0, 1.0, step=0.05,
                key=_EDITOR_KEYS["audio_only_penalty"],
                help=(
                    "Score multiplier for loud moments nothing else corroborates. "
                    "Mechanical noise such as PC fans reads as excitement to the audio detector. "
                    "1.00 applies no penalty."
                ),
            )

        st.subheader("Vertical output", anchor=False)
        st.selectbox(
            "Layout",
            ["source", "letterbox", "crop", "webcam"],
            key=_EDITOR_KEYS["render_layout"],
            help=(
                "source keeps the original aspect. letterbox centres it over a blurred copy "
                "and works on anything. crop takes the centre 9:16 slice. "
                "webcam stacks a cam region above the gameplay."
            ),
        )
        if st.session_state[_EDITOR_KEYS["render_layout"]] == "crop":
            st.caption(
                "A centred 9:16 crop keeps about 32% of a 16:9 width, so minimaps and "
                "kill feeds near the edges are lost. That is usually a feature for short "
                "form, but a persistent overlay can end up sliced in half."
            )
            st.toggle(
                "Use a custom crop region",
                key=_EDITOR_KEYS["crop_enabled"],
                help="Off uses the centred slice. On lets you nudge the window off a stream overlay.",
            )
            if st.session_state[_EDITOR_KEYS["crop_enabled"]]:
                g1, g2, g3, g4 = st.columns(4)
                g1.number_input("x", min_value=0.0, max_value=1.0, step=0.01, key=_EDITOR_KEYS["crop_x"])
                g2.number_input("y", min_value=0.0, max_value=1.0, step=0.01, key=_EDITOR_KEYS["crop_y"])
                g3.number_input("width", min_value=0.01, max_value=1.0, step=0.01, key=_EDITOR_KEYS["crop_w"])
                g4.number_input("height", min_value=0.01, max_value=1.0, step=0.01, key=_EDITOR_KEYS["crop_h"])
                st.caption("Fractions of the source frame. Width 0.316 is exactly 9:16 from 16:9.")

        if st.session_state[_EDITOR_KEYS["render_layout"]] == "webcam":
            st.toggle("Webcam region set", key=_EDITOR_KEYS["webcam_enabled"])
            st.caption(
                "Fractions of the source frame, so one rectangle works for every VOD from "
                "a channel. A stream overlay never moves, so this is drawn once, not detected per frame."
            )
            w1, w2, w3, w4 = st.columns(4)
            w1.number_input("x", min_value=0.0, max_value=1.0, step=0.01, key=_EDITOR_KEYS["webcam_x"])
            w2.number_input("y", min_value=0.0, max_value=1.0, step=0.01, key=_EDITOR_KEYS["webcam_y"])
            w3.number_input("width", min_value=0.01, max_value=1.0, step=0.01, key=_EDITOR_KEYS["webcam_w"])
            w4.number_input("height", min_value=0.01, max_value=1.0, step=0.01, key=_EDITOR_KEYS["webcam_h"])
            st.slider(
                "Webcam share of frame height", 0.05, 0.95, step=0.05,
                key=_EDITOR_KEYS["webcam_fraction"],
            )

        st.subheader("Captions", anchor=False)
        st.toggle(
            "Burn captions into exports",
            key=_EDITOR_KEYS["burn_captions"],
            help="Needs a vertical layout; captions are burned during reframing.",
        )
        c1, c2 = st.columns(2)
        c1.number_input(
            "Caption size", min_value=16, max_value=400, step=4, key=_EDITOR_KEYS["caption_size"]
        )
        c2.toggle("Uppercase captions", key=_EDITOR_KEYS["caption_upper"])
        if (
            st.session_state[_EDITOR_KEYS["burn_captions"]]
            and st.session_state[_EDITOR_KEYS["render_layout"]] == "source"
        ):
            st.warning("Captions need a vertical layout. Pick one above, or exports stay uncaptioned.")

        st.subheader("Chat signal", anchor=False)
        st.caption(
            "The burst detector is relative, so on a quiet chat a single message can look "
            "like a maximum burst. These thresholds require absolute volume before chat counts."
        )
        q1, q2, q3 = st.columns(3)
        q1.number_input(
            "Ignore chat below (msgs/min)", min_value=0.0, max_value=1000.0, step=1.0,
            key=_EDITOR_KEYS["chat_quiet"],
        )
        q2.number_input(
            "Full strength at (msgs/min)", min_value=0.0, max_value=5000.0, step=5.0,
            key=_EDITOR_KEYS["chat_active"],
        )
        q3.number_input(
            "Messages for a full burst", min_value=0.5, max_value=100.0, step=0.5,
            key=_EDITOR_KEYS["chat_burst"],
        )

        st.subheader("Performance", anchor=False)
        st.number_input(
            "Transcription threads (0 = auto)", min_value=0, max_value=256, step=1,
            key=_EDITOR_KEYS["cpu_threads"],
            help=(
                "Auto reserves a core so the desktop stays responsive. Unattended batch runs "
                "override this and use the whole machine."
            ),
        )
        st.slider(
            "Duplicate containment", 0.1, 1.0, step=0.05,
            key=_EDITOR_KEYS["duplicate_containment"],
            help="How much two candidate windows must overlap before one is treated as a duplicate.",
        )

    with reactions:
        st.caption("One phrase per line. These phrases score transcript reactions; changing them can reuse an existing Whisper transcript and simply rescore its text.")
        st.text_area("Reaction phrases", height=420, key=_EDITOR_KEYS["reactions"])

    with transfer:
        st.subheader("Import", anchor=False)
        import_path = path_picker("Settings JSON", "settings_import_path", file_filter=_JSON_FILTER)
        if st.button("Import settings", disabled=not import_path):
            try:
                import_app_settings(import_path, db_path)
                _request_reload("Settings imported into the database. Model download permission and local model selection were not changed.")
                st.rerun()
            except Exception as exc:
                st.exception(exc)

        st.subheader("Export", anchor=False)
        if "settings_export_path" not in st.session_state:
            st.session_state["settings_export_path"] = str(app_root() / "HighlightMiner-settings.json")
        e1, e2 = st.columns([4, 1], vertical_alignment="bottom")
        with e1:
            st.text_input("Export destination", key="settings_export_path")
        with e2:
            st.button("Browse", width="stretch", on_click=_browse_export)
        if st.button("Export settings"):
            try:
                destination = export_app_settings(st.session_state["settings_export_path"], db_path)
                st.success(f"Exported to {destination}. Model download permission and local model selection are intentionally not exported.")
            except Exception as exc:
                st.exception(exc)

    st.divider()
    render_diagnostics_settings(db_path)

    st.divider()
    s1, s2 = st.columns([3, 1])
    with s1:
        if st.button("💾 Save analysis settings", type="primary", width="stretch"):
            try:
                saved = save_app_settings(_build_settings(), db_path)
                _request_reload(f"Settings saved. Weight profile: {detect_weight_preset(saved.weights)}.")
                st.rerun()
            except Exception as exc:
                st.exception(exc)
    with s2:
        if st.button("Reset defaults", width="stretch"):
            reset_app_settings(db_path)
            _request_reload("Settings reset to HighlightMiner defaults. Model download permission and local model selection were left unchanged.")
            st.rerun()
