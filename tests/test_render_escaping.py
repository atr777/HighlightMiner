from __future__ import annotations

from highlightminer.render import Layout, build_filter, escape_filter_path

WINDOWS_PATH = "C:\\work\\clip.ass"
SPACED_PATH = "C:\\vod scrape\\work\\clip.ass"
QUOTED_PATH = "/a/it's.ass"


def test_escapes_and_quotes_a_windows_path():
    assert escape_filter_path(WINDOWS_PATH) == "'C\\:/work/clip.ass'"


def test_quotes_posix_paths_too():
    assert escape_filter_path("/home/a/clip.ass") == "'/home/a/clip.ass'"


def test_paths_with_spaces_survive():
    # The project itself lives under "vod scrape", so this is not hypothetical.
    assert escape_filter_path(SPACED_PATH) == "'C\\:/vod scrape/work/clip.ass'"


def test_apostrophes_are_escaped():
    # A stray quote would otherwise terminate the quoted value early.
    assert escape_filter_path(QUOTED_PATH) == "'/a/it'\\''s.ass'"


def test_backslashes_become_forward_slashes():
    assert "\\\\" not in escape_filter_path(WINDOWS_PATH)


def test_drive_colon_is_escaped():
    # An unescaped colon separates filter options, so ffmpeg would read the rest
    # of the path as the ass filter's original_size argument and fail.
    assert escape_filter_path(WINDOWS_PATH).startswith("'C\\:")


def test_value_is_wrapped_in_single_quotes():
    # Escaping the colon alone is not enough; ffmpeg still splits on it.
    escaped = escape_filter_path(SPACED_PATH)
    assert escaped.startswith("'") and escaped.endswith("'")


def test_build_filter_embeds_the_escaped_path():
    chain = build_filter(Layout(kind="crop"), subtitles=escape_filter_path(SPACED_PATH))
    assert chain.endswith("ass='C\\:/vod scrape/work/clip.ass'")
