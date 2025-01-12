import json
from typing import Tuple

IMG_HEIGHT = IMG_WIDTH = 256

_DATASET_PARAMS = {
    "coco": {"classes_amount": 80},
    "voc": {"classes_amount": 20},
    "default": {"classes_amount": 20}
}


def get_dataset_params(dataset_name: str) -> dict:

    try:
        return _DATASET_PARAMS[dataset_name.lower()]
    except KeyError:
        return _DATASET_PARAMS["default"]


def load_config(filepath: str) -> Tuple[dict, dict, dict]:

    with open(filepath, "r") as f:
        config = json.load(f)

    return config["model"], config["train"], config["data"]