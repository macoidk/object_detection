import os
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision.transforms.v2 as v2
from torchvision.datasets import (CocoDetection, VisionDataset, VOCDetection,
                                  wrap_dataset_for_transforms_v2)

from data.dataset import Dataset
from models.centernet import input_height, input_width
from encoders.centernet_encoder import CenternetEncoder
from utils.io_utils import download_file, unzip_archive


class DataLoader:
    def __init__(self, *, dataset_path: str, image_set: str = None):
        self.image_set = image_set or "train"
        self.dataset_path = dataset_path

    @abstractmethod
    def load(
        self,
        transforms: Optional[Callable] = None,
        encoders: Optional[Callable] = None,
    ) -> Dataset:
        """
        Loads data and returns pytorch VisionDataset.
        The dataset is automatically downloaded
        if `dataset_path` does not exist
        """

    def _post_load_hook(
        self,
        dataset: VisionDataset,
        transforms: Optional[Callable] = None,
        encoder: Optional[Callable] = None,
        *,
        target_keys=None,
    ) -> Dataset:
        transforms = transforms or v2.Compose(
            [
                v2.Resize(size=(input_width, input_height)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        encoder = encoder or CenternetEncoder(input_height, input_width)
        target_keys = target_keys or ["boxes", "labels"]

        adapted_ds = wrap_dataset_for_transforms_v2(dataset, target_keys)
        return Dataset(adapted_ds, transforms, encoder)


class PascalVOCDataLoader(DataLoader):

    def load(
        self,
        transforms: Optional[Callable] = None,
        encoders: Optional[Callable] = None,
    ):
        is_download = not os.path.exists(self.dataset_path)
        raw_ds = VOCDetection(
            root=self.dataset_path,
            year="2007",
            image_set=self.image_set,
            download=is_download,
        )
        return self._post_load_hook(raw_ds, transforms, encoders)


class MSCocoDataLoader(DataLoader):

    DATASET_URLS = {
        "train": {
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "images": "http://images.cocodataset.org/zips/train2017.zip",
            "ann_file": "instances_train2017.json",
        },
        "val": {
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "images": "http://images.cocodataset.org/zips/val2017.zip",
            "ann_file": "instances_val2017.json",
        },
        "test": {
            "annotations": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
            "images": "http://images.cocodataset.org/zips/test2017.zip",
            "ann_file": "image_info_test2017.json",
        },
    }

    def load(
        self,
        transforms: Optional[Callable] = None,
        encoders: Optional[Callable] = None,
    ):
        dataset_config = self.DATASET_URLS[self.image_set]
        ann_folder = Path(self.dataset_path, "annotations")
        images_folder = Path(self.dataset_path, f"{self.image_set}2017")

        # download images and annotations for the dataset
        if not os.path.exists(self.dataset_path):
            ann_folder.mkdir(parents=True)
            images_folder.mkdir(parents=True)

            file_urls = [
                dataset_config["annotations"],
                dataset_config["images"],
            ]
            for url in file_urls:
                filepath = Path(self.dataset_path, url.split("/")[-1])
                download_file(url, filepath)
                unzip_archive(filepath, self.dataset_path)

            print(f"\t\t{self.image_set} dataset is downloaded")

        raw_ds = CocoDetection(
            root=images_folder, annFile=ann_folder / dataset_config["ann_file"]
        )
        return self._post_load_hook(raw_ds, transforms, encoders)


class CustomDataLoader(DataLoader):
    # TODO: implement, find corresponding Detection class
    def load(self):
        pass