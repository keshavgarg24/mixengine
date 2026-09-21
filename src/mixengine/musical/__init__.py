"""
The musicality layer.

Pure logic over the Musical IR: no audio buffers, no file I/O, no model
weights, nothing that needs a GPU. Everything here is a decision a producer
would make, written down so it can be inspected, argued with, and tested.

Keeping this layer free of audio dependencies is deliberate. It means the
engine's musical judgement can be verified on any machine in milliseconds,
independently of whether separation models or stretch libraries are
installed -- and it means a disagreement about what the engine *should*
do is settled by reading one function rather than by rendering a file and
listening.

    theory    harmony, consonance, tuning targets, harmony generation
    groove    microtiming measurement and transfer -- the feel of a track
    salience  where musical accuracy matters, and by how much
    energy    the arc of a song, and how the mix should serve it
"""

from . import energy, groove, salience, theory        # noqa: F401

__all__ = ["theory", "groove", "salience", "energy"]
