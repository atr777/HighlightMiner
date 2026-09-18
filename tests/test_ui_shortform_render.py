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
