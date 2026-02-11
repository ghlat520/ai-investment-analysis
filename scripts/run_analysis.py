#!/usr/bin/env python3
"""
单股分析脚本

用法:
  python scripts/run_analysis.py --stock 000001.SZ
  python scripts/run_analysis.py --stock 600519.SH --market A
"""

import sys
from pathlib import Path

# 确保项目根目录在Python路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "analyze"] + sys.argv[1:]
    main()
