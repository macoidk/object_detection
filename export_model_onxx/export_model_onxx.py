import sys

sys.path.append("..")

from models.centernet import ModelBuilder, IMG_HEIGHT, IMG_WIDTH
from utils.convert_to_onxx import export_to_onnx


def main():
    backbone_name = "resnet50"
    weights = "DEFAULT"

    model = ModelBuilder(
        alpha=1,
        backbone=backbone_name,
        backbone_weights=weights,
    )

    export_to_onnx(
        model=model,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        save_path=f"onnx/model_{backbone_name}",
        batch_size=1,
        show_summary=True,
        test_model=True,
    )


if __name__ == "__main__":
    main()
