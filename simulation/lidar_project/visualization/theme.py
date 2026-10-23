"""Default typography for the report figures, independent of sensor code."""
import json
from pathlib import Path

MATPLOTLIB = json.loads(Path(__file__).with_name("default_theme.json").read_text(encoding="utf-8"))["matplotlib"]
