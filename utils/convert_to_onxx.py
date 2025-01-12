from pathlib import Path

import onnx
import onnxruntime
import torch
import torch.onnx
from torchinfo import summary


def test_onnx_model(onnx_path: str, test_input: torch.Tensor):

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_path)

    ort_inputs = {"input": test_input.detach().cpu().numpy()}

    ort_outputs = ort_session.run(None, ort_inputs)
    print(f"ONNX model tested. output format: {[o.shape for o in ort_outputs]}")
    return ort_outputs


def export_to_onnx(
    model,
    img_height: int,
    img_width: int,
    save_path: str = "models/onnx",
    batch_size: int = 1,
    show_summary: bool = True,
    test_model: bool = True,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if show_summary:
        summary(model, (batch_size, 3, img_height, img_width))

    x = torch.randn(batch_size, 3, img_height, img_width, requires_grad=True).to(device)

    with torch.no_grad():
        outputs = model(x)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    backbone_name = model.backbone_name if hasattr(model, "backbone_name") else "custom"
    onnx_path = save_dir / f"centernet_{backbone_name}.onnx"

    if isinstance(outputs, dict):
        output_names = list(outputs.keys())
    else:
        output_names = ["output"]

    # Експортуємо модель
    torch.onnx.export(
        model,
        x,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            **{name: {0: "batch_size"} for name in output_names},
        },
    )

    print(f"model exported {onnx_path}")

    if test_model:
        try:
            test_onnx_model(onnx_path, x)
        except Exception as e:
            print(f"error while test {str(e)}")

    return onnx_path
