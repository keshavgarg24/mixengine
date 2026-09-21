# RECONSTRUCTED after the project folder was lost (2026-10-07). The original
# bytes were not in any backup or transcript; this is the minimal file the rest
# of the code needs.
"""`python -m mixengine` runs the command line."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
