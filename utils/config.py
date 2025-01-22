import json
from datetime import datetime
from typing import Tuple

IMG_HEIGHT = IMG_WIDTH = 256
TENSORBOARD_FOLDER = f"runs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_config(filepath: str) -> Tuple[dict, dict, dict]:
    """
    Returns configs for model, training and data
    """
    with open(filepath, "r") as f:
        config = json.load(f)

    return config["model"], config["train"], config["data"]
