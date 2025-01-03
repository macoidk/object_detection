import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision.transforms.v2 as v2
from torchvision.datasets import CocoDetection, VisionDataset, VOCDetection
from torchvision.transforms.v2 import Transform

from data.dataset import Dataset
from models.centernet import input_height, input_width
from encoders.centernet_encoder import CenternetEncoder
from utils.io_utils import download_file, unzip_archive


class DataLoader(ABC):
    def __init__(self, *, dataset_path: str, image_set: str = None):
        self.image_set = image_set or "train"
        self.dataset_path = dataset_path

    @abstractmethod
    def load(
        self,
        transforms: Optional[Transform] = None,
        encoder: Optional[Callable] = None,
    ) -> Dataset:
        """
        Loads data and returns pytorch VisionDataset.
        The dataset is automatically downloaded
        if `dataset_path` does not exist
        """

    def _post_load_hook(
        self,
        dataset: VisionDataset,
        transforms: Optional[Transform] = None,
        encoder: Optional[Callable] = None,
        *,
        target_keys=None,
    ) -> Dataset:
        transforms = transforms or v2.Compose(
            [
                v2.Resize(size=(input_width, input_height), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        encoder = encoder or CenternetEncoder(input_height, input_width)
        return Dataset(dataset, transforms, encoder)


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

    def _convert_coco_annotations(self, target):
        """Converts COCO annotations to VOC format"""
        if not isinstance(target, list):
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64)
            }

        boxes = []
        labels = []

        for annotation in target:
            if not isinstance(annotation, dict):
                continue

            bbox = annotation.get('bbox')
            category_id = annotation.get('category_id')

            if bbox is None or category_id is None:
                continue

            try:
                # COCO format: [x, y, width, height] to [x1, y1, x2, y2]
                x1 = float(bbox[0])
                y1 = float(bbox[1])
                w = float(bbox[2])
                h = float(bbox[3])
                x2 = x1 + w
                y2 = y1 + h

                if w <= 0 or h <= 0:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(category_id)
            except (ValueError, TypeError, IndexError) as e:
                print(f"Error processing bbox {bbox}: {e}")
                continue

        if not boxes:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64)
            }

        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

    def load(
            self,
            transforms: Optional[Transform] = None,
            encoder: Optional[Callable] = None,
    ):
        dataset_config = self.DATASET_URLS[self.image_set]
        ann_folder = Path(self.dataset_path) / "annotations"
        images_folder = Path(self.dataset_path) / f"{self.image_set}2017"

        if not os.path.exists(self.dataset_path):
            os.makedirs(ann_folder, exist_ok=True)
            os.makedirs(images_folder, exist_ok=True)

            file_urls = [
                dataset_config["annotations"],
                dataset_config["images"],
            ]
            for url in file_urls:
                filepath = Path(self.dataset_path) / url.split("/")[-1]
                download_file(url, filepath)
                unzip_archive(filepath, self.dataset_path)

            print(f"\t\t{self.image_set} dataset is downloaded")

        class CocoDetectionTransformed(CocoDetection):
            def __init__(self, root, annFile, transform_fn):
                super().__init__(root, annFile)
                self.transform_fn = transform_fn

            def __getitem__(self, index):
                img, target = super().__getitem__(index)
                return img, self.transform_fn(target)

        raw_ds = CocoDetectionTransformed(
            root=str(images_folder),
            annFile=str(ann_folder / dataset_config["ann_file"]),
            transform_fn=self._convert_coco_annotations
        )

        return self._post_load_hook(raw_ds, transforms, encoder)