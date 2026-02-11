#!/usr/bin/env python3
"""
LLM成本报告脚本

用法:
  python scripts/cost_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "cost"]
    main()
