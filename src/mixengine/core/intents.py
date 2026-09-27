"""
What the user tells the engine.

Measurement can establish that a vocal sits within 14 cents of a
semitone grid. It cannot establish that the singer meant it, or that
this beat is the one they recorded against. Those are facts about
provenance, and the only reliable source for them is the person who
made the recording.

Every field defaults to None, meaning "decide for me". A field that is
set wins outright. This mirrors what the reference tools do -- RoEx
requires stem role tags, Neutron asks which track is the focus, Nectar
makes key a manual field -- and it exists because the alternative,
guessing, is what produced a render that retuned a finished vocal
toward the wrong key.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Dict, Optional

VOCAL_STATES = ("raw", "tuned", "finished")
RELATIONSHIPS = ("locked", "free")
SPACES = ("keep", "match", "add")
SEPARATIONS = ("never", "always")
PERFORMANCES = ("rap", "melodic_rap", "sung", "spoken")
LEAD_INS = ("trim", "keep")
ENTRIES = ("section", "top")
NOISE_ANSWERS = ("accept",)


def _choice(value: Any, allowed: tuple, field_name: str) -> Optional[str]:
    if value is None or value == "auto" or value == "":
        return None
    text = str(value).strip().lower()
    if text not in allowed:
        raise ValueError(
            f"{field_name} must be one of {', '.join(('auto',) + allowed)}, "
            f"got {value!r}")
    return text


def _strength(value: Any, field_name: str) -> Optional[float]:
    """A strength is `auto`, `off`, or a number in [0, 1]."""
    if value is None or value == "auto" or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be auto, off, or 0-1, got {value!r}")
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "off":
            return 0.0
        try:
            value = float(text)
        except ValueError:
            raise ValueError(
                f"{field_name} must be auto, off, or 0-1, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be auto, off, or 0-1, got {value!r}")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be within 0-1, got {number}")
    return number


def _number(value: Any, field_name: str,
            low: float, high: float) -> Optional[float]:
    if value is None or value == "auto" or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number, got {value!r}")
    if not low <= number <= high:
        raise ValueError(
            f"{field_name} must be within {low}-{high}, got {number}")
    return number


@dataclass(frozen=True)
class Intents:
    """User-stated facts. None means "decide for me"."""

    vocal_state: Optional[str] = None      # raw | tuned | finished
    relationship: Optional[str] = None     # locked | free
    key: Optional[str] = None              # e.g. "f_minor"
    bpm: Optional[float] = None
    tune: Optional[float] = None           # 0 = off
    timing: Optional[float] = None         # 0 = off
    space: Optional[str] = None            # keep | match | add
    separate: Optional[str] = None         # never | always
    loudness: Optional[float] = None       # target LUFS
    nudge: Optional[float] = None          # beats; +1 = a beat later
    # Answers to what the analysis could not settle on its own. Each is
    # asked only when the measurement was unsure; see core/questions.py.
    performance: Optional[str] = None      # rap | melodic_rap | sung | spoken
    lead_in: Optional[str] = None          # trim | keep the sound before line 1
    entry: Optional[str] = None            # section: come in at the drop | top
    noise: Optional[str] = None            # accept: render a severe take anyway

    AUTO: ClassVar["Intents"]

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "Intents":
        if not d:
            return Intents()
        key = d.get("key")
        return Intents(
            vocal_state=_choice(d.get("vocal_state"), VOCAL_STATES,
                                "vocal_state"),
            relationship=_choice(d.get("relationship"), RELATIONSHIPS,
                                 "relationship"),
            key=(str(key).strip().lower()
                 if key not in (None, "", "auto") else None),
            bpm=_number(d.get("bpm"), "bpm", 40.0, 220.0),
            tune=_strength(d.get("tune"), "tune"),
            timing=_strength(d.get("timing"), "timing"),
            space=_choice(d.get("space"), SPACES, "space"),
            separate=_choice(d.get("separate"), SEPARATIONS, "separate"),
            loudness=_number(d.get("loudness"), "loudness", -30.0, -3.0),
            nudge=_number(d.get("nudge"), "nudge", -16.0, 16.0),
            performance=_choice(d.get("performance"), PERFORMANCES,
                                "performance"),
            lead_in=_choice(d.get("lead_in"), LEAD_INS, "lead_in"),
            entry=_choice(d.get("entry"), ENTRIES, "entry"),
            noise=_choice(d.get("noise"), NOISE_ANSWERS, "noise"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_all_auto(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))


Intents.AUTO = Intents()
