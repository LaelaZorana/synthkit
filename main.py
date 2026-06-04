#!/usr/bin/env python3
"""Zero-install entry point: `python3 main.py text gen --demo`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthkit.cli import main

if __name__ == "__main__":
    main()
