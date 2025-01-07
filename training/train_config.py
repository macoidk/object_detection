import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from callbacks.tensorboard_callback import TensorBoardCallback
from data.data_loader import MSCocoDataLoader, PascalVOCDataLoader
from encoders.mscoco_encoder import MSCOCOCenternetEncoder
from encoders.centernet_encoder import CenternetEncoder
from models.centernet import ModelBuilder, IMG_WIDTH, IMG_HEIGHT
from utils.config import load_config, get_dataset_params
from map_calculator import MAPCalculator


def get_dataset_loader(dataset_name: str):
    """Factory function to get the appropriate dataset loader"""
    loaders = {
        "coco": MSCocoDataLoader,
        "voc": PascalVOCDataLoader
    }
    return loaders.get(dataset_name.lower())


def get_dataset_encoder(dataset_name: str, img_height: int, img_width: int, down_ratio: int):
    """Factory function to get the appropriate encoder"""
    if dataset_name.lower() == "coco":
        return MSCOCOCenternetEncoder(
            img_height=img_height,
            img_width=img_width,
            down_ratio=down_ratio,
            n_classes=80
        )
    else:  # VOC
        return CenternetEncoder(
            img_height=img_height,
            img_width=img_width,
            down_ratio=down_ratio
        )


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
    dataset = kwargs.get("dataset", "unknown")
    cur_dir = Path(__file__).resolve().parent

    checkpoint_filename = (
            cur_dir.parent / checkpoints_dir / f"pretrained_weights_{tag}_{backbone}_{dataset}.pt"
    )

    torch.save(model.state_dict(), checkpoint_filename)
    print(f"Saved model checkpoint to {checkpoint_filename}")


def evaluate_model(model, dataloader, device, map_calculator=None):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data in dataloader:
            input_data, gt_data = data
            input_data = input_data.to(device).contiguous()
            gt_data = gt_data.to(device)

            predictions = model(input_data)
            loss_dict = model(input_data, gt=gt_data)

            if map_calculator is not None:
                map_calculator.update(predictions, gt_data)

            total_loss += loss_dict["loss"].item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    map_score = map_calculator.compute() if map_calculator is not None else 0.0
    map_calculator.reset() if map_calculator is not None else None

    return avg_loss, map_score


def train(model_conf, train_conf, data_conf):
    tensorboard = TensorBoardCallback()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get appropriate dataset loader class
    dataset_config = data_conf["dataset"]
    DatasetLoader = get_dataset_loader(dataset_config["name"])

    if DatasetLoader is None:
        raise ValueError(f"Unknown dataset: {dataset_config['name']}")

    # Initialize data loaders
    train_loader = DatasetLoader(
        dataset_path=dataset_config["path"],
        image_set=dataset_config["train_set"]
    )
    val_loader = DatasetLoader(
        dataset_path=dataset_config["path"],
        image_set=dataset_config["val_set"]
    )

    transform = transforms.Compose([
        transforms.Resize(size=(IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    # Get appropriate encoder
    encoder = get_dataset_encoder(
        dataset_config["name"],
        img_height=256,
        img_width=256,
        down_ratio=4
    )

    if train_conf["is_overfit"]:
        val_dataset = val_loader.load(transform, encoder)
        training_data = torch.utils.data.Subset(val_dataset, range(train_conf["subset_len"]))
        batch_size = train_conf["subset_len"]
        tag = "overfit"
        print(f"Running in overfit mode with subset of {dataset_config['name']} validation data")
    else:
        training_data = train_loader.load(transform, encoder)
        validation_data = val_loader.load(transform, encoder)
        batch_size = train_conf["batch_size"]
        tag = "train"
        print(f"Running in training mode with full {dataset_config['name']} train/val split")

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=not train_conf["is_overfit"],
        num_workers=0
    )

    if not train_conf["is_overfit"]:
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    num_classes = get_dataset_params(dataset_config["name"])["classes_amount"]
    map_calculator = MAPCalculator(num_classes=num_classes)

    model = ModelBuilder(
        alpha=model_conf["alpha"],
        backbone=model_conf["backbone"]["name"],
        backbone_weights=model_conf["backbone"]["pretrained_weights"],
        class_number=num_classes

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

            # Get predictions for mAP calculation
            predictions = model(input_data)
            map_calculator.update(predictions, gt_data)

            loss_dict = model(input_data, gt=gt_data)
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()

            curr_lr = optimizer.param_groups[0]['lr']

            if train_conf["is_overfit"]:
                map_score = map_calculator.compute()
                map_calculator.reset()
                batch_metrics = {
                    "val_loss": loss_dict["loss"].item(),
                    "map": map_score
                }
                print(
                    f"Epoch {epoch}, batch {i}, val_loss={loss_dict['loss'].item():.3f}, mAP={map_score:.3f}, lr={curr_lr}")
            else:
                batch_metrics = {"train_loss": loss_dict["loss"].item()}
                print(f"Epoch {epoch}, batch {i}, train_loss={loss_dict['loss'].item():.3f}, lr={curr_lr}")

            tensorboard.log_batch(epoch, i, batch_metrics, curr_lr)

            train_loss += loss_dict["loss"].item()
            num_train_batches += 1

        avg_loss = train_loss / num_train_batches if num_train_batches > 0 else float('inf')

        if train_conf["is_overfit"]:
            map_score = map_calculator.compute()
            map_calculator.reset()
            epoch_loss_dict = {
                "val_loss": avg_loss,
                "map": map_score
            }
            print(f"Epoch {epoch}: val_loss={avg_loss:.3f}, mAP={map_score:.3f}")
            scheduler.step(avg_loss)
        else:
            val_loss, map_score = evaluate_model(model, val_loader, device, map_calculator)
            epoch_loss_dict = {
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "map": map_score
            }
            print(f"Epoch {epoch}: train_loss={avg_loss:.3f}, val_loss={val_loss:.3f}, mAP={map_score:.3f}")
            scheduler.step(val_loss)

        tensorboard.on_epoch_end(epoch, epoch_loss_dict, curr_lr)

        if criteria_satisfied(avg_loss, epoch):
            break

        epoch += 1

    tensorboard.close()

    save_model(
        model,
        model_conf["weights_path"],
        tag=tag,
        backbone=model_conf["backbone"]["name"],
        dataset=dataset_config["name"]
    )


def main(config_path: str = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to config file")
    args = parser.parse_args()

    filepath = args.config or config_path

    model_conf, train_conf, data_conf = load_config(filepath)

    train(model_conf, train_conf, data_conf)


if __name__ == "__main__":
    main()