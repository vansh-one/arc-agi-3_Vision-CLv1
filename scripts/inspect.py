#!/usr/bin/env python3
"""Convenience wrapper for inspecting weights."""
from __future__ import annotations

import sys

from vision_arc_agi.cli import inspect_main

if __name__ == "__main__":
    sys.exit(inspect_main(sys.argv[1:]))
