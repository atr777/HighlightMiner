from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .security import validate_settings_file
from .util import load_json

_STANDARD_WHISPER_MODELS = {
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3",
}
_ALLOWED_DEVICES = {"auto", "cpu", "cuda"}
_ALLOWED_COMPUTE_TYPES = {
    "auto", "int8", "int8_float16", "int8_float32", "int8_bfloat16",
    "float16", "float32", "bfloat16",
}


@dataclass
class Settings:
    whisper_model: str = "large-v3"
    allow_custom_whisper_model: bool = False
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    audio_window_sec: float = 1.0
    audio_hop_sec: float = 0.5
    pre_roll_sec: float = 18.0
    post_roll_sec: float = 14.0
    merge_gap_sec: float = 10.0
    max_candidate_sec: float = 75.0
    min_candidate_score: float = 0.38
    max_candidates: int = 40
    chat_min_burst_messages: float = 3.0
    chat_quiet_msgs_per_min: float = 15.0
    chat_active_msgs_per_min: float = 60.0
    weights: dict[str, float] = field(default_factory=lambda: {
        "audio": 0.34, "transcript": 0.42, "chat": 0.24
    })
    reaction_phrases: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_file(cls, path: str | Path | None) -> "Settings":
        if path is None:
            return cls()
        validated = validate_settings_file(path)
        data = load_json(validated)
        if not isinstance(data, dict):
            raise ValueError("settings.json must contain a JSON object.")
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def validate(self) -> None:
        self.whisper_model = str(self.whisper_model).strip()
        if not self.whisper_model:
            raise ValueError("whisper_model cannot be empty.")
        if not self.allow_custom_whisper_model and self.whisper_model not in _STANDARD_WHISPER_MODELS:
            raise ValueError(
                f"Custom Whisper model {self.whisper_model!r} is blocked by default. "
                "Use a standard model name or explicitly set allow_custom_whisper_model=true."
            )

        self.device = str(self.device).lower().strip()
        if self.device not in _ALLOWED_DEVICES:
            raise ValueError(f"device must be one of {sorted(_ALLOWED_DEVICES)}")
        self.compute_type = str(self.compute_type).lower().strip()
        if self.compute_type not in _ALLOWED_COMPUTE_TYPES:
            raise ValueError(f"compute_type must be one of {sorted(_ALLOWED_COMPUTE_TYPES)}")

        if self.language is not None:
            self.language = str(self.language).strip() or None
            if self.language and len(self.language) > 32:
                raise ValueError("language is unexpectedly long.")

        self.beam_size = int(self.beam_size)
        self.max_candidates = int(self.max_candidates)
        if not 1 <= self.beam_size <= 20:
            raise ValueError("beam_size must be between 1 and 20.")
        if not 1 <= self.max_candidates <= 500:
            raise ValueError("max_candidates must be between 1 and 500.")

        ranges = {
            "audio_window_sec": (0.1, 10.0),
            "audio_hop_sec": (0.05, 5.0),
            "pre_roll_sec": (0.0, 600.0),
            "post_roll_sec": (0.0, 600.0),
            "merge_gap_sec": (0.0, 600.0),
            "max_candidate_sec": (1.0, 1800.0),
            "min_candidate_score": (0.0, 1.0),
            "chat_min_burst_messages": (0.5, 100.0),
            "chat_quiet_msgs_per_min": (0.0, 1000.0),
            "chat_active_msgs_per_min": (0.0, 5000.0),
        }
        for name, (low, high) in ranges.items():
            value = float(getattr(self, name))
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}.")
            setattr(self, name, value)

        if self.chat_active_msgs_per_min < self.chat_quiet_msgs_per_min:
            raise ValueError(
                "chat_active_msgs_per_min must be greater than or equal to chat_quiet_msgs_per_min."
            )

        if not isinstance(self.weights, dict):
            raise ValueError("weights must be a JSON object.")
        cleaned_weights: dict[str, float] = {}
        for key in ("audio", "transcript", "chat"):
            value = float(self.weights.get(key, 0.0))
            if not 0.0 <= value <= 10.0:
                raise ValueError(f"weights.{key} must be between 0 and 10.")
            cleaned_weights[key] = value
        if sum(cleaned_weights.values()) <= 0:
            raise ValueError("At least one scoring weight must be greater than zero.")
        self.weights = cleaned_weights

        if not isinstance(self.reaction_phrases, list):
            raise ValueError("reaction_phrases must be a JSON array.")
        if len(self.reaction_phrases) > 1000:
            raise ValueError("reaction_phrases contains too many entries (maximum 1000).")
        phrases: list[str] = []
        for phrase in self.reaction_phrases:
            text = str(phrase).strip()
            if not text:
                continue
            if len(text) > 200:
                raise ValueError("A reaction phrase is longer than 200 characters.")
            phrases.append(text)
        self.reaction_phrases = phrases

    def normalized_weights(
        self,
        chat_available: bool,
        transcript_available: bool = True,
    ) -> dict[str, float]:
        weights = dict(self.weights)
        available = {"audio": True, "transcript": transcript_available, "chat": chat_available}
        for key, is_available in available.items():
            if not is_available:
                weights[key] = 0.0

        cleaned = {key: max(0.0, float(value)) for key, value in weights.items()}
        total = sum(cleaned.values())
        if total <= 0.0:
            active = [key for key, is_available in available.items() if is_available]
            equal = 1.0 / len(active)
            return {key: equal if key in active else 0.0 for key in weights}
        return {key: value / total for key, value in cleaned.items()}
