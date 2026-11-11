#!/usr/bin/env python3
"""Placeholder for a make target that is not built yet. Always fails, loudly."""

import sys

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "?"
    gate = sys.argv[2] if len(sys.argv) > 2 else "?"
    print(f"NOT IMPLEMENTED: make {target} (needed by {gate})", file=sys.stderr)
    sys.exit(1)
