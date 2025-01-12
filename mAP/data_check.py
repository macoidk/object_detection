import os
import json
from pathlib import Path
import defusedxml.ElementTree as ET
from typing import Dict, List


class GTDataChecker:
    def __init__(self, voc_root: str = "../VOC"):
        self.voc_path = Path(voc_root) / "VOCdevkit" / "VOC2007"
        self.ann_path = self.voc_path / "Annotations"

    def load_voc_gt(self) -> Dict[str, List]:
        """Завантаження GT даних з VOC формату"""
        gt_boxes = {}

        for xml_file in self.ann_path.glob("*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Отримуємо image_id з імені файлу
            image_id = xml_file.stem
            boxes = []

            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                # Перевіряємо валідність боксу
                if xmin >= xmax or ymin >= ymax:
                    print(f"Невалідний бокс в {image_id}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                    continue

                boxes.append([xmin, ymin, xmax, ymax])

            if boxes:  # Додаємо тільки якщо є валідні бокси
                gt_boxes[image_id] = boxes

        return gt_boxes

    def load_coco_gt(self, coco_file: str) -> Dict[str, List]:
        """Завантаження GT даних з COCO формату"""
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        gt_boxes = {}

        # Створюємо маппінг image_id -> filename
        image_map = {img['id']: img['file_name'] for img in coco_data['images']}

        for ann in coco_data['annotations']:
            image_id = str(ann['image_id'])
            bbox = ann['bbox']

            # Конвертуємо з COCO [x, y, width, height] в [xmin, ymin, xmax, ymax]
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]

            if image_id not in gt_boxes:
                gt_boxes[image_id] = []

            gt_boxes[image_id].append([xmin, ymin, xmax, ymax])

        return gt_boxes

    def verify_gt_format(self, gt_boxes: Dict[str, List]):
        """Перевірка формату GT даних"""
        print(f"Всього зображень: {len(gt_boxes)}")

        for image_id, boxes in gt_boxes.items():
            # Перевірка формату image_id
            if not isinstance(image_id, str):
                print(f"Неправильний формат image_id: {image_id}")

            # Перевірка списку боксів
            if not isinstance(boxes, list):
                print(f"Неправильний формат boxes для {image_id}")
                continue

            # Перевірка кожного боксу
            for box in boxes:
                if not isinstance(box, list) or len(box) != 4:
                    print(f"Неправильний формат боксу в {image_id}: {box}")
                    continue

                xmin, ymin, xmax, ymax = box
                if xmin >= xmax or ymin >= ymax:
                    print(f"Невалідний бокс в {image_id}: {box}")

        print("Перевірка завершена")


def main():
    checker = GTDataChecker()

    # Завантажуємо GT дані з VOC
    print("Завантаження GT даних з VOC...")
    gt_boxes_voc = checker.load_voc_gt()
    print("Перевірка VOC GT даних...")
    checker.verify_gt_format(gt_boxes_voc)

    # Завантажуємо GT дані з COCO
    coco_file = checker.ann_path / "trainval_cocoformat.json"
    if coco_file.exists():
        print("\nЗавантаження GT даних з COCO...")
        gt_boxes_coco = checker.load_coco_gt(str(coco_file))
        print("Перевірка COCO GT даних...")
        checker.verify_gt_format(gt_boxes_coco)

    # Приклад як передавати дані в evaluator
    print("\nПриклад даних для evaluator.py:")
    image_id = list(gt_boxes_voc.keys())[0]
    print(f"GT boxes для {image_id}:")
    print(gt_boxes_voc[image_id])


if __name__ == "__main__":
    main()