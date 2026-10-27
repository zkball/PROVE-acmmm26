"""python -m prove  →  scripts/infer.py"""

from pathlib import Path
import runpy
import sys

_script = Path(__file__).resolve().parent.parent / "scripts" / "infer.py"
sys.argv[0] = str(_script)
runpy.run_path(str(_script), run_name="__main__")
