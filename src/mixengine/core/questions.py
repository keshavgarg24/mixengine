"""
What the engine asks before it renders.

Measurement settles most things. What it cannot settle it should not
guess at silently: whether a loud blob before the first line is a
count-in to cut or an intro to keep, whether a take with no sustained
pitch is rap or the tracker losing a voice under noise, whether a take
whose noise the chain could not remove should be rendered at all. Each
of those is a fact the person who made the recording has and the engine
does not, so each becomes a question -- asked only when the analysis was
unsure, with the engine's own answer as the default, and answered
through an `Intents` field so the render reads it like any other stated
fact.

Severity says what the interface does with it. `info` and `warn` are
shown with their defaults and a render goes ahead if nobody answers;
`block` stops the render until it is answered, because the default
answer is "do not render this".
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SEVERITIES = ("info", "warn", "block")

# The performance classifier below this is asked rather than trusted.
PERFORMANCE_ASK_BELOW = 0.7
# A key or tempo the detector was this unsure of is confirmed.
KEY_ASK_BELOW = 0.5
BPM_ASK_BELOW = 0.4

PERFORMANCE_LABELS = {"rap": "rap", "melodic_rap": "melodic rap",
                      "sung": "singing", "spoken": "spoken word"}


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    options: List[Dict[str, str]]    # [{"value", "label"}]
    default: str
    reason: str
    severity: str = "info"
    intent: str = ""                 # the Intents field the answer sets
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _opt(value: str, label: str) -> Dict[str, str]:
    return {"value": value, "label": label}


def _noise_question(vdna: dict) -> Optional[Question]:
    noise = vdna.get("noise") or {}
    verdict = noise.get("verdict")
    if verdict == "severe":
        return Question(
            id="noise", severity="block", intent="noise",
            text=("The background noise in this take is nearly as loud as "
                  "the voice. Restoration took it down but could not remove "
                  "it, and it will be heard under every line. A cleaner "
                  "take is the fix."),
            options=[_opt("rerecord", "I'll upload a cleaner take"),
                     _opt("accept", "Render this one anyway")],
            default="rerecord",
            reason=("the words clear the noise by %.0f dB after restoration "
                    "(%.0f dB before); a mix wants 30 or more"
                    % (noise.get("snr_db") or 0.0,
                       noise.get("input_snr_db") or 0.0)),
            detail=noise)
    if verdict == "heavy":
        return Question(
            id="noise", severity="warn", intent="noise",
            text=("This take carries heavy background noise. Restoration "
                  "took most of it down; some will remain under the words."),
            options=[_opt("accept", "Render it"),
                     _opt("rerecord", "I'll upload a cleaner take")],
            default="accept",
            reason=("the words clear the noise by %.0f dB after restoration"
                    % (noise.get("snr_db") or 0.0)),
            detail=noise)
    return None


def _start_question(vdna: dict) -> Optional[Question]:
    span = vdna.get("performance_span") or {}
    lead, tail = span.get("lead_in_s") or 0.0, span.get("tail_s") or 0.0
    if lead <= 0 and tail <= 0:
        return None
    where = []
    if lead > 0:
        where.append("%.1f s of sound before the first line" % lead)
    if tail > 0:
        where.append("%.1f s after the last" % tail)
    default = span.get("default") or "keep"
    return Question(
        id="start", intent="lead_in",
        severity="warn" if default == "trim" else "info",
        text=("The take has %s (the performance runs %.1f-%.1f s). Cut it, "
              "so the first line lands where the beat arrives, or keep it?"
              % (" and ".join(where), span.get("start_s") or 0.0,
                 span.get("end_s") or 0.0)),
        options=[_opt("trim", "Cut it -- start at the first line"),
                 _opt("keep", "Keep it -- it is part of the take")],
        default=default, reason=span.get("reason") or "", detail=span)


def _performance_question(vdna: dict) -> Optional[Question]:
    label = vdna.get("performance_type") or "sung"
    conf = vdna.get("performance_confidence")
    if conf is None or float(conf) >= PERFORMANCE_ASK_BELOW:
        return None
    conf = float(conf)
    return Question(
        id="performance", intent="performance", severity="warn",
        text=("This reads as %s, but not clearly. Rap is left untuned and "
              "tightened to the grid; singing is tuned and its timing kept "
              "loose. Which is it?" % PERFORMANCE_LABELS.get(label, label)),
        options=[_opt(v, PERFORMANCE_LABELS[v]) for v in
                 ("rap", "melodic_rap", "sung", "spoken")],
        default=label, reason=vdna.get("performance_reason") or "",
        detail={"confidence": conf})


def _key_question(vdna: dict) -> Optional[Question]:
    key = vdna.get("key") or {}
    conf = float(vdna.get("key_confidence") or 0.0)
    name = key.get("name")
    if not name or conf >= KEY_ASK_BELOW:
        return None
    options = [_opt(name, name)]
    for cand in (vdna.get("key_candidates") or [])[:3]:
        # candidates are (name, score) pairs; a dict form is tolerated
        cname = (cand.get("name") if isinstance(cand, dict)
                 else cand[0] if isinstance(cand, (list, tuple)) and cand
                 else None)
        if cname and cname != name and len(options) < 3:
            options.append(_opt(cname, cname))
    options.append(_opt("auto", "Not sure -- work it out"))
    return Question(
        id="key", intent="key", severity="info",
        text="The key reads as %s, but the detector is not sure. Is that "
             "right?" % name,
        options=options, default="auto",
        reason="key confidence %.0f%%" % (conf * 100),
        detail={"confidence": conf})


def _bpm_question(vdna: dict) -> Optional[Question]:
    bpm = float(vdna.get("bpm") or 0.0)
    conf = float(vdna.get("bpm_confidence") or 0.0)
    src = vdna.get("bpm_source")
    if src == "user" or (bpm > 0 and conf >= BPM_ASK_BELOW):
        return None
    options = []
    if bpm > 0:
        options.append(_opt("%.1f" % bpm, "%.0f BPM" % bpm))
    for alt in (vdna.get("bpm_alternates") or [])[:2]:
        try:
            value = float(alt.get("bpm") or 0 if isinstance(alt, dict) else alt)
        except (TypeError, ValueError):
            continue
        if value > 0 and abs(value - bpm) > 0.5:
            options.append(_opt("%.1f" % value, "%.0f BPM" % value))
    if not options:
        return None
    options.append(_opt("auto", "Not sure -- work it out"))
    return Question(
        id="bpm", intent="bpm", severity="info",
        text=("The take's tempo is not clear from its onsets%s. Do you "
              "know it?" % (" (best guess %.0f BPM)" % bpm if bpm else "")),
        options=options, default="auto",
        reason="tempo confidence %.0f%%" % (conf * 100),
        detail={"confidence": conf, "source": src})


def _entry_question(bdna: Optional[dict]) -> Optional[Question]:
    if not bdna:
        return None
    sections = bdna.get("sections") or []
    if not sections:
        return None
    first = sections[0]
    label = str(first.get("label") or "").lower()
    if label not in ("intro", "break"):
        return None
    bars = first.get("bars") or first.get("duration_bars")
    if not bars:
        try:
            bar_s = float(bdna.get("bar_s") or 0.0)
            bars = round((float(first.get("end_s") or first.get("end") or 0.0)
                          - float(first.get("start_s") or first.get("start") or 0.0))
                         / bar_s) if bar_s > 0 else 0
        except (TypeError, ValueError):
            bars = 0
    if not bars or bars > 8:
        return None
    return Question(
        id="entry", intent="entry", severity="info",
        text=("The beat opens with a %d-bar %s. Bring the vocal in when the "
              "beat arrives, or from the top?" % (int(bars), label)),
        options=[_opt("section", "When the beat arrives"),
                 _opt("top", "From the top")],
        default="section",
        reason="a vocal left at the top of the beat sits inside its %s" % label)


def questions_for(vdna: dict, bdna: Optional[dict] = None) -> List[Question]:
    """Everything the analysis could not settle, most serious first."""
    out = [q for q in (_noise_question(vdna), _start_question(vdna),
                       _performance_question(vdna), _key_question(vdna),
                       _bpm_question(vdna), _entry_question(bdna)) if q]
    order = {"block": 0, "warn": 1, "info": 2}
    out.sort(key=lambda q: order.get(q.severity, 3))
    return out


def unanswered_blocks(questions: List[Question],
                      intents: Optional[Any]) -> List[Question]:
    """Blocking questions whose intent the caller has not set."""
    return [q for q in questions
            if q.severity == "block"
            and getattr(intents, q.intent, None) in (None, "")]
