"""
Vocal capture: measuring the input, and coaching the performance.

Split the same way the rest of the engine is -- measurement separated from
judgement, and both separated from I/O:

    realtime   streaming frame analysis and MPM pitch tracking
    coach      what to tell the singer, when, and when to stay quiet

Neither module opens an audio device. They take numpy blocks and return
numbers and cues, which keeps the whole capture path testable without a
microphone and leaves the choice of backend -- PortAudio, CoreAudio, a
browser worklet -- to the host application.
"""

from . import coach, realtime          # noqa: F401

__all__ = ["realtime", "coach"]
