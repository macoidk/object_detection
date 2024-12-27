import argparse
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils import data

from callbacks.model_save import SaveBestModelCallback
from data.dataset import Dataset
from models.imagenet_eiuo import ImageNetModel, input_height, input_width
from encoders.imagenet_encoder import ImageNetEncoder

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overfit", action="store_true", help="overfit to 10 images")
args = parser.parse_args()

overfit = args.overfit
image_set = "val" if overfit else "train"

dataset_val = torchvision.datasets.VOCDetection(
    root="../VOC", year="2007", image_set=image_set, download=True
)

transform = transforms.Compose(
    [
        transforms.Resize(size=(input_width, input_height)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

encoder = ImageNetEncoder(
    img_height=input_height,
    img_width=input_width,
    down_ratio=4,
    n_classes=20
)
dataset_val = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset_val)
torch_dataset = Dataset(dataset=dataset_val, transformation=transform, encoder=encoder)

training_data = torch_dataset
lr = 0.03
batch_size = 32
patience = 7
min_lr = 1e-3

def criteria_satisfied(_, current_epoch):
    if current_epoch >= 10000:
        return True
    return False

if overfit:
    subset_len = 10
    training_data = torch.utils.data.Subset(torch_dataset, range(subset_len))
    batch_size = subset_len
    lr = 5e-2
    patience = 50
    min_lr = 1e-3

    def criteria_satisfied(current_loss, _):
        if current_loss < 1.0:
            return True
        return False

print(f"Selected image_set: {image_set}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageNetModel(alpha=1.0, class_number=20, down_ratio=4).to(device)

save_callback = SaveBestModelCallback(
    save_dir="../callbacks/imagenet_checkpoints",
    metric_name="loss",
    greater_is_better=False,
    start_saving_threshold=5.0,
    min_improvement=0.5,
)

parameters = list(model.parameters())
optimizer = torch.optim.Adam(parameters, lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,
    patience=patience,
    threshold=1e-4,
    threshold_mode="rel",
    cooldown=1,
    min_lr=min_lr,
)

model.train(True)

batch_generator = torch.utils.data.DataLoader(
    training_data, num_workers=0, batch_size=batch_size, shuffle=False
)

epoch = 1
get_desired_loss = False

while True:
    print(f"EPOCH {epoch}:")
    epoch_loss = 0
    batch_count = 0

    for _, data in enumerate(batch_generator):
        input_data, gt_data = data
        input_data = input_data.to(device).contiguous()

        gt_data = gt_data.to(device)
        gt_data.requires_grad = False

        loss_dict = model(input_data, gt=gt_data)
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        epoch_loss += loss_dict["loss"].item()
        batch_count += 1

        save_callback.on_eval_epoch_end(
            model=model, optimizer=optimizer, epoch=epoch, current_metric=loss_dict["loss"].item()
        )

    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / batch_count
    current_lr = optimizer.param_groups[0]['lr']

    if criteria_satisfied(loss_dict["loss"], epoch):
        break

    scheduler.step(avg_epoch_loss)  # Use average epoch loss for scheduler
    epoch += 1

torch.save(model.state_dict(), "../models/checkpoints/imagenet_pretrained_weights.pt")