#!/usr/bin/env python3
"""Pre-cache the ARC-AGI-3 public game catalogue locally so subsequent
training passes can run fully OFFLINE."""
from __future__ import annotations

import sys

from vision_arc_agi.cli import download_main

if __name__ == "__main__":
    sys.exit(download_main(sys.argv[1:]))
