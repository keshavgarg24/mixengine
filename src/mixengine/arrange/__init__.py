"""
Arrangement: turning a vocal over a beat into a song with a shape.

The engine could already mix. What it could not do was *arrange* -- decide
that bar 33 is a hook and should be doubled, that the verse before it
should sit back so the hook has somewhere to arrive from, that the last
line wants a harmony and the gaps want ad-libs. Without that, every bar
gets identical treatment, and a mix of uniform bars is a demo however
clean it is.

  `structure`  finds the song inside the take: which phrases repeat, which
               one is the hook, where the bridge is.
  `plan`       turns that into an energy arc and, from the arc, the two
               concrete outputs: which layers exist where, and how the mix
               moves over time.
  `layers`     generates the layers themselves -- doubles, octaves,
               harmony, ad-libs -- from the single take available.
  `transitions` renders the section-change devices the plan asked for:
               risers, impacts, reversed tails, dropouts.
"""

from .layers import Layer, build as build_layers, sum_layers
from .plan import SongPlan, build as build_plan
from .structure import StructureResult, analyze as analyze_structure
from .transitions import build as build_transitions

__all__ = [
    "Layer", "SongPlan", "StructureResult",
    "analyze_structure", "build_layers", "build_plan", "build_transitions",
    "sum_layers",
]
