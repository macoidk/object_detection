import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms

from callbacks.tensorboard_callback import TensorBoardCallback
from data.data_loader import PascalVOCDataLoader
from encoders.centernet_encoder import CenternetEncoder
from models.centernet import IMG_HEIGHT, IMG_WIDTH, ModelBuilder
from utils.config import load_config
from utils.evaluator import get_avg_precision_at_iou


def convert_predictions(model_output):
    print("Debugging model_output:")
    print("Type of model_output:", type(model_output))
    print("Keys in model_output:", model_output.keys() if isinstance(model_output, dict) else "Not a dictionary")
    print("Full model_output content:", model_output)

    # Оригінальний код
    batch_size = model_output['heatmap'].shape[0]
    predictions = []

    for b in range(batch_size):
        img_preds = []
        for c in range(model_output['heatmap'].shape[1]):
            heatmap = model_output['heatmap'][b, c]
            reg_x = model_output['reg_x'][b, c]
            reg_y = model_output['reg_y'][b, c]
            reg_w = model_output['reg_w'][b, c]
            reg_h = model_output['reg_h'][b, c]

            peaks = find_peaks(heatmap)
            for peak in peaks:
                score = heatmap[peak[0], peak[1]]
                if score > DETECTION_THRESHOLD:
                    x = (peak[1] + reg_x[peak[0], peak[1]]) * OUTPUT_STRIDE
                    y = (peak[0] + reg_y[peak[0], peak[1]]) * OUTPUT_STRIDE
                    w = reg_w[peak[0], peak[1]] * IMG_WIDTH
                    h = reg_h[peak[0], peak[1]] * IMG_HEIGHT

                    xmin = max(0, x - w / 2)
                    ymin = max(0, y - h / 2)
                    xmax = min(IMG_WIDTH, x + w / 2)
                    ymax = min(IMG_HEIGHT, y + h / 2)

                    img_preds.append({
                        'bbox': [xmin, ymin, xmax, ymax],
                        'score': score.item(),
                        'category_id': c + 1
                    })
        predictions.append(img_preds)

    return predictions


def convert_gt_data(gt_data):
    """Конвертує ground truth дані у формат для обчислення mAP"""
    gt_boxes = {}

    for i in range(len(gt_data)):
        # Отримуємо boxes з ground truth даних
        # Тут структура залежить від вашого encoder
        if isinstance(gt_data[i], dict):
            boxes = gt_data[i].get('boxes', [])
        else:
            # Якщо gt_data не словник, перевіряємо інші можливі формати
            boxes = gt_data[i] if isinstance(gt_data[i], torch.Tensor) else []

        # Конвертуємо в numpy та список
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy().tolist()

        gt_boxes[str(i)] = {'boxes': boxes}

    return gt_boxes


def criteria_builder(stop_loss, stop_epoch):
    def criteria_satisfied(current_loss, current_epoch):
        if stop_loss is not None and current_loss < stop_loss:
            return True
        if stop_epoch is not None and current_epoch > stop_epoch:
            return True
        return False

    return criteria_satisfied


def save_model(model, weights_path: str = None, **kwargs):
    checkpoints_dir = weights_path or "models/checkpoints"
    tag = kwargs.get("tag", "train")
    backbone = kwargs.get("backbone", "default")
    cur_dir = Path(__file__).resolve().parent

    checkpoint_filename = (
            cur_dir.parent / checkpoints_dir / f"pretrained_weights_{tag}_{backbone}.pt"
    )

    torch.save(model.state_dict(), checkpoint_filename)
    print(f"Збережено чекпоінт моделі: {checkpoint_filename}")


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = {}
    all_gt = {}

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            input_data, gt_data = data
            input_data = input_data.to(device).contiguous()
            gt_data = gt_data.to(device)

            # Отримуємо передбачення моделі
            output = model(input_data, gt=gt_data)

            print("Model output keys:", output.keys())
            print("GT data structure:", type(gt_data), gt_data[0].keys() if isinstance(gt_data, (list, tuple)) else gt_data.keys())


            total_loss += output["loss"].item()

            # Конвертуємо gt та передбачення
            batch_predictions = convert_predictions(output)
            batch_gt = convert_gt_data(gt_data)



            # Зберігаємо ground truth та передбачення
            for i in range(len(gt_data)):
                img_id = f"{batch_idx}_{i}"
                all_predictions[img_id] = batch_predictions[str(i)]
                all_gt[img_id] = batch_gt[str(i)]

            num_batches += 1

    # Обчислюємо mAP
    map_results = get_avg_precision_at_iou(all_gt, all_predictions, iou_thr=0.5)
    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    return {
        'loss': avg_loss,
        'mAP': map_results['avg_prec']
    }


def calculate_batch_map(model_output, gt_data):
    """Обчислює mAP для одного батчу"""
    predictions = convert_predictions(model_output)
    gt_boxes = convert_gt_data(gt_data)
    map_results = get_avg_precision_at_iou(gt_boxes, predictions, iou_thr=0.5)
    return map_results['avg_prec']


def train(model_conf, train_conf, data_conf):
    tensorboard = TensorBoardCallback()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = PascalVOCDataLoader(dataset_path="../VOC", image_set="train")
    val_loader = PascalVOCDataLoader(dataset_path="../VOC", image_set="val")

    transform = transforms.Compose(
        [
            transforms.Resize(size=(IMG_WIDTH, IMG_HEIGHT)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    encoder = CenternetEncoder(IMG_HEIGHT, IMG_WIDTH)

    if train_conf["is_overfit"]:
        val_dataset = val_loader.load(transform, encoder)
        training_data = torch.utils.data.Subset(
            val_dataset, range(train_conf["subset_len"])
        )
        batch_size = train_conf["subset_len"]
        tag = "overfit"
        print("Запуск у режимі оверфіту на підмножині валідаційних даних")
    else:
        training_data = train_loader.load(transform, encoder)
        validation_data = val_loader.load(transform, encoder)
        batch_size = train_conf["batch_size"]
        tag = "train"
        print("Запуск у режимі тренування на повному наборі даних")

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=not train_conf["is_overfit"],
        num_workers=0,
    )

    if not train_conf["is_overfit"]:
        val_loader = torch.utils.data.DataLoader(
            validation_data, batch_size=batch_size, shuffle=False, num_workers=0
        )

    model = ModelBuilder(
        alpha=model_conf["alpha"],
        backbone=model_conf["backbone"]["name"],
        backbone_weights=model_conf["backbone"]["pretrained_weights"],
    ).to(device)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=train_conf["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_conf["lr_schedule"]["factor"],
        patience=train_conf["lr_schedule"]["patience"],
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=1,
        min_lr=train_conf["lr_schedule"]["min_lr"],
    )

    criteria_satisfied = criteria_builder(*train_conf["stop_criteria"].values())

    epoch = 1
    while True:
        model.train()
        train_loss = 0
        num_train_batches = 0

        for i, data in enumerate(train_loader):
            input_data, gt_data = data
            input_data = input_data.to(device).contiguous()
            gt_data = gt_data.to(device)
            gt_data.requires_grad = False

            model_output = model(input_data, gt=gt_data)
            optimizer.zero_grad()
            model_output["loss"].backward()
            optimizer.step()

            curr_lr = scheduler.get_last_lr()[0]

            if train_conf["is_overfit"]:
                # Обчислюємо mAP для кожного батчу у режимі оверфіту
                batch_map = calculate_batch_map(model_output, gt_data)
                batch_metrics = {
                    "val_loss": model_output["loss"].item(),
                    "val_mAP": batch_map
                }
                print(
                    f"Epoch {epoch}, batch {i}, "
                    f"val_loss={model_output['loss'].item():.3f}, "
                    f"val_mAP={batch_map:.3f}, "
                    f"lr={curr_lr}"
                )
            else:
                batch_metrics = {"train_loss": model_output["loss"].item()}
                print(
                    f"Epoch {epoch}, batch {i}, "
                    f"train_loss={model_output['loss'].item():.3f}, "
                    f"lr={curr_lr}"
                )

            tensorboard.log_batch(epoch, i, batch_metrics, curr_lr)
            train_loss += model_output["loss"].item()
            num_train_batches += 1

        avg_loss = train_loss / num_train_batches if num_train_batches > 0 else float("inf")

        if train_conf["is_overfit"]:
            eval_results = evaluate_model(model, train_loader, device)
            epoch_metrics = {
                "val_loss": eval_results['loss'],
                "val_mAP": eval_results['mAP']
            }
            print(
                f"Epoch {epoch}: "
                f"val_loss={eval_results['loss']:.3f}, "
                f"val_mAP={eval_results['mAP']:.3f}"
            )
            scheduler.step(eval_results['loss'])
        else:
            eval_results = evaluate_model(model, val_loader, device)
            epoch_metrics = {
                "train_loss": avg_loss,
                "val_loss": eval_results['loss'],
                "val_mAP": eval_results['mAP']
            }
            print(
                f"Epoch {epoch}: "
                f"train_loss={avg_loss:.3f}, "
                f"val_loss={eval_results['loss']:.3f}, "
                f"val_mAP={eval_results['mAP']:.3f}"
            )
            scheduler.step(eval_results['loss'])

        tensorboard.on_epoch_end(epoch, epoch_metrics, curr_lr)

        if criteria_satisfied(avg_loss, epoch):
            break

        epoch += 1

    tensorboard.close()

    save_model(
        model,
        model_conf["weights_path"],
        tag=tag,
        backbone=model_conf["backbone"]["name"],
    )


def main(config_path: str = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="шлях до конфігураційного файлу")
    args = parser.parse_args()

    filepath = args.config or config_path

    model_conf, train_conf, data_conf = load_config(filepath)

    train(model_conf, train_conf, data_conf)


if __name__ == "__main__":
    main()