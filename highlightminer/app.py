from __future__ import annotations

import logging

import streamlit as st

from highlightminer.analysis_jobs import recover_stale_analysis_jobs
from highlightminer.database_diagnostics import initialize_database_with_diagnostics
from highlightminer.diagnostics import log_event, log_exception, log_startup
from highlightminer.storage import default_db_path
from highlightminer.shutdown import active_work_shutdown_block_reason
from highlightminer.ui_common import render_shutdown
from highlightminer.ui_mine import analysis_is_running, render_mine_page
from highlightminer.ui_crop import render_crop_page
from highlightminer.ui_settings import render_settings_page
from highlightminer.ui_style import apply_shell_style

_NAV_ITEMS = ["⛏️ Mine / Review", "🎯 Crop", "⚙️ Settings"]
_NAV_KEY = "main_navigation"


def _render_app() -> None:
    st.set_page_config(page_title="HighlightMiner", page_icon="⛏️", layout="wide")
    apply_shell_style()
    db_path = default_db_path()
    initialize_database_with_diagnostics(db_path)
    recovered_jobs = recover_stale_analysis_jobs(db_path)
    if recovered_jobs:
        log_event(
            "analysis.stale_jobs_recovered",
            level=logging.WARNING,
            job_ids=recovered_jobs,
        )
        st.session_state.setdefault(
            "analysis_error",
            f"Recovered {len(recovered_jobs)} analysis job(s) whose worker heartbeat expired.",
        )
    running = analysis_is_running(db_path)

    with st.container(border=True):
        st.title("⛏️ HighlightMiner", anchor=False)
        st.caption(
            "Mine long VODs for the moments worth keeping — audio + optional local speech recognition + optional chat, "
            "ranked locally on your machine. v0.2 keeps source-aware history, reviews, and app settings in SQLite."
        )

    if running:
        st.session_state[_NAV_KEY] = _NAV_ITEMS[0]

    with st.sidebar:
        st.header("HighlightMiner", anchor=False)
        page = st.radio(
            "Navigate",
            _NAV_ITEMS,
            key=_NAV_KEY,
            label_visibility="collapsed",
            disabled=running,
        )
        if running:
            st.caption(
                "🔒 Analysis controls are locked until the active run finishes, fails, "
                "or is safely cancelled while waiting for input."
            )

    if running:
        page = _NAV_ITEMS[0]

    if page == "⚙️ Settings":
        with st.sidebar:
            st.caption(f"Database: `{db_path}`")
            render_shutdown(block_reason=active_work_shutdown_block_reason(db_path))
        render_settings_page(db_path)
        return

    if page == "🎯 Crop":
        with st.sidebar:
            st.caption(f"Database: `{db_path}`")
            render_shutdown(block_reason=active_work_shutdown_block_reason(db_path))
        render_crop_page(db_path)
        return

    render_mine_page(db_path)


def main() -> None:
    log_startup(entrypoint="streamlit")
    try:
        _render_app()
    except Exception as exc:
        log_exception("app.unhandled_error", exc)
        raise


if __name__ == "__main__":
    main()
