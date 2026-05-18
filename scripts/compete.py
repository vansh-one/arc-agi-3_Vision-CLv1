#!/usr/bin/env python3
"""Convenience wrapper: ``python scripts/compete.py [flags]`` → vision-arc-compete."""
from __future__ import annotations

import sys

from vision_arc_agi.cli import compete_main

if __name__ == "__main__":
    sys.exit(compete_main(sys.argv[1:]))
