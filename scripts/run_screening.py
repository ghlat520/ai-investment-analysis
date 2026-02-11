#!/usr/bin/env python3
"""
量化筛选脚本

用法:
  python scripts/run_screening.py
  python scripts/run_screening.py --top-n 20
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "screen"] + sys.argv[1:]
    main()
