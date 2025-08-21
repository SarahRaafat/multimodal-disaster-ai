# tests/conftest.py
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Add project root and ./src so "import src...." works
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
