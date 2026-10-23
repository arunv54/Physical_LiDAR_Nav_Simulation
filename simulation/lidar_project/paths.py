"""Project locations shared by command-line adapters, never by sensor physics."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SENSOR_CONFIG = CONFIG_DIR / "sensor.jsonc"
DEFAULT_TRAJECTORY_CONFIG = CONFIG_DIR / "trajectory.jsonc"
