"""Render the pages the short-form work touches.

Importing a Streamlit module proves almost nothing: widget key collisions,
missing session state and bad selectbox options only surface when the script
actually runs. These render the real pages.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _run(script: str, tmp_path: Path):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(script)
    app.run(timeout=60)
    return app


def test_settings_page_renders(tmp_path):
    db = (tmp_path / "highlightminer.db").as_posix()
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_settings import render_settings_page
render_settings_page(Path(r"{db}"))
''',
        tmp_path,
    )
    assert not app.exception, app.exception


def test_settings_page_exposes_the_short_form_tab(tmp_path):
    db = (tmp_path / "highlightminer.db").as_posix()
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_settings import render_settings_page
render_settings_page(Path(r"{db}"))
''',
        tmp_path,
    )
    assert not app.exception
    labels = {t.label for t in app.selectbox}
    assert "Timing preset" in labels
    assert "Layout" in labels


def test_settings_page_renders_with_short_form_active(tmp_path):
    """The webcam branch adds widgets that only appear for that layout."""
    from highlightminer.config import Settings
    from highlightminer.settings_store import save_app_settings

    db = tmp_path / "highlightminer.db"
    save_app_settings(
        Settings(
            short_form_mode=True,
            render_layout="webcam",
            webcam_rect={"x": 0.7, "y": 0.0, "w": 0.3, "h": 0.3},
            burn_captions=True,
        ),
        db,
    )
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_settings import render_settings_page
render_settings_page(Path(r"{db.as_posix()}"))
''',
        tmp_path,
    )
    assert not app.exception, app.exception


def test_settings_page_warns_about_captions_without_a_layout(tmp_path):
    from highlightminer.config import Settings
    from highlightminer.settings_store import save_app_settings

    db = tmp_path / "highlightminer.db"
    save_app_settings(Settings(burn_captions=True, render_layout="source"), db)
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_settings import render_settings_page
render_settings_page(Path(r"{db.as_posix()}"))
''',
        tmp_path,
    )
    assert not app.exception
    assert any("vertical layout" in w.value for w in app.warning)


def test_mine_page_renders_with_the_url_ingest_panel(tmp_path):
    db = (tmp_path / "highlightminer.db").as_posix()
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_mine import render_mine_page
render_mine_page(Path(r"{db}"))
''',
        tmp_path,
    )
    assert not app.exception, app.exception


def test_crop_page_renders(tmp_path):
    db = (tmp_path / "highlightminer.db").as_posix()
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_crop import render_crop_page
render_crop_page(Path(r"{db}"))
''',
        tmp_path,
    )
    assert not app.exception, app.exception


def test_crop_page_warns_when_the_layout_is_not_crop(tmp_path):
    from highlightminer.config import Settings
    from highlightminer.settings_store import save_app_settings

    db = tmp_path / "highlightminer.db"
    save_app_settings(Settings(render_layout="letterbox"), db)
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_crop import render_crop_page
render_crop_page(Path(r"{db.as_posix()}"))
''',
        tmp_path,
    )
    assert not app.exception
    assert any("letterbox" in i.value for i in app.info)


def test_crop_page_seeds_sliders_from_the_saved_region(tmp_path):
    from highlightminer.config import Settings
    from highlightminer.settings_store import save_app_settings

    db = tmp_path / "highlightminer.db"
    save_app_settings(
        Settings(render_layout="crop", gameplay_rect={"x": 0.27, "y": 0.0, "w": 0.316406, "h": 1.0}),
        db,
    )
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_crop import render_crop_page
render_crop_page(Path(r"{db.as_posix()}"))
''',
        tmp_path,
    )
    assert not app.exception
    # The position sliders only render once frames have been sampled, so the
    # saved region shows up in seeded state rather than in a widget.
    assert app.session_state["crop_tool_x"] == pytest.approx(0.27)
    assert app.session_state["crop_tool_h"] == pytest.approx(1.0)


def _tiny_video(path: Path) -> Path:
    """A real, probe-able video. The review page legitimately reads its duration."""
    import subprocess

    from highlightminer.media import require_executable

    subprocess.run(
        [
            require_executable("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=navy:s=320x180:d=2",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            str(path),
        ],
        check=True, capture_output=True,
    )
    return path


def _analysis_db(tmp_path, render_layout="crop"):
    """A database with one stored analysis, for driving the review page."""
    from dataclasses import asdict

    from highlightminer.config import Settings
    from highlightminer.settings_store import save_app_settings
    from highlightminer.storage import save_analysis

    db = tmp_path / "highlightminer.db"
    video = _tiny_video(tmp_path / "v.mp4")
    settings = Settings(render_layout=render_layout, short_form_mode=True)
    save_app_settings(settings, db)
    payload = {
        "version": 1,
        "video_path": str(video),
        "content_label": "T",
        "duration": 2.0,
        "media": {"duration": 2.0, "streams": []},
        "transcription": {"language": "en"},
        "chat": {"path": None, "messages": 0},
        "settings": asdict(settings),
        "work_dir": str(tmp_path / "work"),
        "candidates": [{
            "id": "H001", "rank": 1, "score": 0.9, "peak_time": 1.0,
            "start": 0.2, "end": 1.8, "start_label": "00:00", "end_label": "00:01",
            "audio_score": 0.9, "transcript_score": 0.5, "chat_score": 0.0,
            "reason": "audio spike", "transcript": "words", "content_label": "T",
            "features": {},
        }],
    }
    analysis_id = save_analysis(db, payload, [], [], [], work_dir=str(tmp_path / "work"))
    return db, analysis_id


def test_review_page_renders_with_a_stored_analysis(tmp_path):
    db, analysis_id = _analysis_db(tmp_path)
    app = _run(
        f'''
import streamlit as st
from pathlib import Path
from highlightminer.ui_mine import render_mine_page
st.session_state["analysis_id"] = "{analysis_id}"
render_mine_page(Path(r"{db.as_posix()}"))
''',
        tmp_path,
    )
    assert not app.exception, app.exception


def test_review_page_has_no_candidate_dropdown(tmp_path):
    """The ranked table is the selector; a second control would duplicate it."""
    db, analysis_id = _analysis_db(tmp_path)
    app = _run(
        f'''
import streamlit as st
from pathlib import Path
from highlightminer.ui_mine import render_mine_page
st.session_state["analysis_id"] = "{analysis_id}"
render_mine_page(Path(r"{db.as_posix()}"))
''',
        tmp_path,
    )
    assert not app.exception
    assert not any(w.label == "Review candidate" for w in app.selectbox)


def test_crop_controls_explain_themselves_on_a_non_crop_layout(tmp_path):
    db, analysis_id = _analysis_db(tmp_path, render_layout="letterbox")
    app = _run(
        f'''
import streamlit as st
from pathlib import Path
from highlightminer.ui_mine import render_mine_page
st.session_state["analysis_id"] = "{analysis_id}"
render_mine_page(Path(r"{db.as_posix()}"))
''',
        tmp_path,
    )
    assert not app.exception


def test_settings_page_exposes_the_transcription_engine(tmp_path):
    db = (tmp_path / "highlightminer.db").as_posix()
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_settings import render_settings_page
render_settings_page(Path(r"{db}"))
''',
        tmp_path,
    )
    assert not app.exception, app.exception
    assert "Engine" in {t.label for t in app.selectbox}


def test_the_engine_page_reports_the_vulkan_build_state(tmp_path):
    """Either it found the build or it says what is missing; never silence."""
    from highlightminer.config import Settings
    from highlightminer.settings_store import save_app_settings

    db = tmp_path / "highlightminer.db"
    save_app_settings(Settings(transcription_backend="whispercpp"), db)
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_settings import render_settings_page
render_settings_page(Path(r"{db.as_posix()}"))
''',
        tmp_path,
    )
    assert not app.exception, app.exception
    assert app.success or app.warning


def test_turning_refinement_off_on_the_gpu_engine_warns(tmp_path):
    from highlightminer.config import Settings
    from highlightminer.settings_store import save_app_settings

    db = tmp_path / "highlightminer.db"
    save_app_settings(
        Settings(transcription_backend="whispercpp", refine_word_timings=False), db
    )
    app = _run(
        f'''
from pathlib import Path
from highlightminer.ui_settings import render_settings_page
render_settings_page(Path(r"{db.as_posix()}"))
''',
        tmp_path,
    )
    assert not app.exception, app.exception
    assert any("word timings" in w.value for w in app.warning)
