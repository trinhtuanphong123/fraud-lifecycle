from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
