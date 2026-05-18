#!/usr/bin/env python3
"""Convenience wrapper: ``python scripts/train.py [flags]`` → vision-arc-train."""
from __future__ import annotations

import sys

from vision_arc_agi.cli import train_main

if __name__ == "__main__":
    sys.exit(train_main(sys.argv[1:]))
