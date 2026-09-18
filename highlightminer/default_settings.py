from __future__ import annotations

from .config import Settings

_DEFAULT_REACTION_PHRASES = [
    "what the fuck", "what the hell", "no way", "holy shit", "oh my god",
    "are you kidding", "you've got to be kidding", "i can't believe", "that worked",
    "what is happening", "what just happened", "fuck me", "oh fuck", "shit",
    "mitä vittua", "mitä helvettiä", "ei saatana", "voi vittu", "ei jumalauta",
    "mitä tapahtuu", "mitä tapahtui", "ei voi olla",
]


def product_default_settings() -> Settings:
    """Return a fresh canonical HighlightMiner settings profile.

    This is intentionally code-defined so the in-app Reset defaults action is
    not affected if a user edits the legacy/interchange settings.json file.
    """
    return Settings(
        whisper_model="large-v3",
        allow_custom_whisper_model=False,
        device="auto",
        compute_type="auto",
        language=None,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        audio_window_sec=1.0,
        audio_hop_sec=0.5,
        pre_roll_sec=18.0,
        post_roll_sec=14.0,
        merge_gap_sec=10.0,
        max_candidate_sec=75.0,
        min_candidate_score=0.38,
        max_candidates=40,
        short_form_mode=False,
        render_layout="source",
        burn_captions=False,
        caption_font_size=96,
        caption_uppercase=False,
        webcam_fraction=0.3,
        webcam_rect=None,
        gameplay_rect=None,
        audio_only_penalty=1.0,
        duplicate_containment=0.65,
        cpu_threads=0,
        hook_lead_sec=3.0,
        min_candidate_sec=8.0,
        speech_snap_sec=1.5,
        chat_min_burst_messages=3.0,
        chat_quiet_msgs_per_min=15.0,
        chat_active_msgs_per_min=60.0,
        weights={"audio": 0.34, "transcript": 0.42, "chat": 0.24},
        reaction_phrases=list(_DEFAULT_REACTION_PHRASES),
    )
