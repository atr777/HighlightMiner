from __future__ import annotations

import base64
import os
import subprocess
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import streamlit as st

_VIDEO_FILTER = "Video files|*.mp4;*.mkv;*.mov;*.webm;*.avi;*.m4v;*.ts|All files|*.*"
_CHAT_FILTER = "Chat files|*.json;*.jsonl;*.ndjson;*.csv|All files|*.*"
_JSON_FILTER = "JSON files|*.json|All files|*.*"
_PERSISTED_WIDGET_PREFIX = "_hm_persisted_widget:"


WORK_DIR_ENV = "HIGHLIGHTMINER_WORK"


def default_work_dir() -> str:
    """Where scratch files go, unless told otherwise.

    Mirrors HIGHLIGHTMINER_DB. Working files are large, so they usually belong
    on a different drive from the application, and without this the folder had
    to be retyped in the sidebar on every launch. Getting it wrong once meant a
    13 GB download landing on the system drive.
    """
    from .runtime import app_root

    override = os.environ.get(WORK_DIR_ENV, "").strip()
    if override:
        return str(Path(override).expanduser())
    return str(app_root() / "highlightminer_work")


def persisted_widget_key(state_key: str) -> str:
    return f"{_PERSISTED_WIDGET_PREFIX}{state_key}"


def hydrate_persistent_widget(
    state: MutableMapping[str, Any],
    state_key: str,
    default: Any = "",
) -> None:
    """Restore a widget value after Streamlit cleaned up an unrendered widget key."""
    backing_key = persisted_widget_key(state_key)
    if backing_key not in state:
        state[backing_key] = state.get(state_key, default)
    if state_key not in state:
        state[state_key] = state[backing_key]


def persist_widget_value(state: MutableMapping[str, Any], state_key: str) -> None:
    backing_key = persisted_widget_key(state_key)
    state[backing_key] = state.get(state_key, "")


def dialog_initial_directory(value: str | None) -> str:
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            candidate = candidate.parent
        elif not candidate.is_dir():
            candidate = candidate.parent
        if candidate.is_dir():
            return str(candidate.resolve())
    return str(Path.home())


def _ps_literal(value: str) -> str:
    return value.replace("'", "''")


def _run_windows_dialog(script: str) -> str | None:
    if os.name != "nt":
        raise RuntimeError("Native Browse buttons are currently available on Windows only.")
    wrapped = "$ErrorActionPreference='Stop';[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false);" + script
    encoded = base64.b64encode(wrapped.encode("utf-16le")).decode("ascii")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-EncodedCommand", encoded],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    stdout = result.stdout.decode("utf-8-sig", errors="replace").strip()
    stderr = result.stderr.decode("utf-8-sig", errors="replace").strip()
    if result.returncode != 0:
        raise RuntimeError(stderr or f"Native file dialog failed with exit code {result.returncode}.")
    return stdout or None


def choose_file(title: str, file_filter: str, initial: str | None = None) -> str | None:
    script = f"""
Add-Type -AssemblyName System.Windows.Forms;
$d=New-Object System.Windows.Forms.OpenFileDialog;
$d.Title='{_ps_literal(title)}';$d.Filter='{_ps_literal(file_filter)}';
$d.InitialDirectory='{_ps_literal(dialog_initial_directory(initial))}';
$d.CheckFileExists=$true;$d.Multiselect=$false;$d.RestoreDirectory=$true;
if($d.ShowDialog()-eq [System.Windows.Forms.DialogResult]::OK){{[Console]::Out.Write($d.FileName)}};$d.Dispose();
"""
    return _run_windows_dialog(script)


def choose_folder(title: str, initial: str | None = None) -> str | None:
    script = f"""
Add-Type -AssemblyName System.Windows.Forms;
$d=New-Object System.Windows.Forms.FolderBrowserDialog;
$d.Description='{_ps_literal(title)}';$d.SelectedPath='{_ps_literal(dialog_initial_directory(initial))}';
$d.ShowNewFolderButton=$true;
if($d.ShowDialog()-eq [System.Windows.Forms.DialogResult]::OK){{[Console]::Out.Write($d.SelectedPath)}};$d.Dispose();
"""
    return _run_windows_dialog(script)


def choose_save_file(title: str, initial: str | None = None) -> str | None:
    initial_path = Path(initial).expanduser() if initial else Path.home() / "HighlightMiner-settings.json"
    script = f"""
Add-Type -AssemblyName System.Windows.Forms;
$d=New-Object System.Windows.Forms.SaveFileDialog;
$d.Title='{_ps_literal(title)}';$d.Filter='JSON files|*.json|All files|*.*';
$d.InitialDirectory='{_ps_literal(dialog_initial_directory(str(initial_path)))}';
$d.FileName='{_ps_literal(initial_path.name)}';$d.DefaultExt='json';$d.AddExtension=$true;$d.OverwritePrompt=$true;
if($d.ShowDialog()-eq [System.Windows.Forms.DialogResult]::OK){{[Console]::Out.Write($d.FileName)}};$d.Dispose();
"""
    return _run_windows_dialog(script)


def browse_into_state(state_key: str, title: str, *, folder: bool = False, file_filter: str = "All files|*.*") -> None:
    try:
        current = st.session_state.get(state_key, "")
        selected = choose_folder(title, current) if folder else choose_file(title, file_filter, current)
        if selected:
            st.session_state[state_key] = selected
            persist_widget_value(st.session_state, state_key)
    except Exception as exc:
        st.session_state["native_dialog_error"] = str(exc)


def path_picker(
    label: str,
    state_key: str,
    *,
    default: str = "",
    placeholder: str | None = None,
    folder: bool = False,
    file_filter: str = "All files|*.*",
    disabled: bool = False,
) -> str:
    hydrate_persistent_widget(st.session_state, state_key, default)
    path_col, browse_col = st.columns([4, 1], vertical_alignment="bottom")
    with path_col:
        st.text_input(label, key=state_key, placeholder=placeholder, disabled=disabled)
    with browse_col:
        st.button(
            "Browse",
            key=f"browse_{state_key}",
            width="stretch",
            on_click=browse_into_state,
            args=(state_key, f"Choose {label}"),
            kwargs={"folder": folder, "file_filter": file_filter},
            disabled=disabled,
        )
    persist_widget_value(st.session_state, state_key)
    return str(st.session_state.get(state_key, ""))


def persistent_text_input(
    label: str,
    state_key: str,
    *,
    default: str = "",
    placeholder: str | None = None,
    help: str | None = None,
    disabled: bool = False,
) -> str:
    hydrate_persistent_widget(st.session_state, state_key, default)
    st.text_input(label, key=state_key, placeholder=placeholder, help=help, disabled=disabled)
    persist_widget_value(st.session_state, state_key)
    return str(st.session_state.get(state_key, ""))


def render_shutdown(*, block_reason: str | None = None) -> None:
    shutdown_file = os.environ.get("HIGHLIGHTMINER_SHUTDOWN_FILE")
    if not shutdown_file:
        return
    st.divider()
    if st.button(
        "🛑 Exit HighlightMiner",
        width="stretch",
        disabled=bool(block_reason),
    ):
        try:
            Path(shutdown_file).write_text("shutdown\n", encoding="utf-8")
        except OSError as exc:
            st.error(f"Could not request shutdown: {exc}")
        else:
            st.info("Shutting down HighlightMiner…")
            st.stop()
    if block_reason:
        st.caption(f"🔒 {block_reason}")
