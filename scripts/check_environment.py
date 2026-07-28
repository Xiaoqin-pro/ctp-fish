from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.reproducibility import environment_summary
print(json.dumps(environment_summary(), indent=2))
