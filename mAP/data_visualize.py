import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from PIL import Image
import defusedxml.ElementTree as ET
from utils.voc2coco import dump_voc_classes, get_label2id, convert_xmls_to_cocojson


class DatasetVisualizer:
    def __init__(self, voc_root="../VOC"):
        self.voc_root = Path(voc_root)
        self.voc_path = self.voc_root / "VOCdevkit" / "VOC2007"
        self.ann_path = self.voc_path / "Annotations"
        self.img_path = self.voc_path / "JPEGImages"
        self.labels_path = self.voc_path / "labels.txt"

        # Перевіряємо наявність датасету
        if not self.voc_path.exists():
            raise ValueError(f"Датасет не знайдено за шляхом: {self.voc_path}")

        # Отримуємо класи
        self.classes = dump_voc_classes(str(self.ann_path))
        self.label2id = get_label2id(str(self.labels_path))

    def load_voc_annotation(self, xml_path):
        """Завантаження анотації у форматі VOC"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)

        return boxes, labels

    def load_coco_annotation(self, json_path, image_id):
        """Завантаження анотації у форматі COCO"""
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        boxes = []
        labels = []

        # Знаходимо всі анотації для конкретного зображення
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                box = ann['bbox']
                # Конвертуємо з [x, y, width, height] в [xmin, ymin, xmax, ymax]
                boxes.append([
                    box[0],
                    box[1],
                    box[0] + box[2],
                    box[1] + box[3]
                ])

                # Знаходимо назву класу за category_id
                category = next(cat for cat in coco_data['categories']
                                if cat['id'] == ann['category_id'])
                labels.append(category['name'])

        return boxes, labels

    def visualize_boxes(self, image_path, boxes, labels, title=""):
        """Візуалізація боксів на зображенні"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(img)

        # Малюємо бокси
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            plt.gca().add_patch(rect)
            plt.text(
                xmin, ymin - 5,
                label,
                color='white',
                bbox=dict(facecolor='red', alpha=0.5)
            )

        plt.title(title)
        plt.axis('off')
        plt.show()

    def compare_annotations(self, image_name):
        """Порівняння анотацій у форматах VOC та COCO"""
        # Завантажуємо VOC анотацію
        xml_path = self.ann_path / f"{image_name}.xml"
        voc_boxes, voc_labels = self.load_voc_annotation(xml_path)

        # Завантажуємо COCO анотацію
        json_path = self.ann_path / "trainval_cocoformat.json"
        image_id = int(''.join(filter(str.isdigit, image_name)))
        coco_boxes, coco_labels = self.load_coco_annotation(json_path, image_id)

        # Візуалізуємо обидва формати
        img_path = self.img_path / f"{image_name}.jpg"

        plt.figure(figsize=(20, 8))

        # VOC
        plt.subplot(1, 2, 1)
        self.visualize_boxes(img_path, voc_boxes, voc_labels, "VOC формат")

        # COCO
        plt.subplot(1, 2, 2)
        self.visualize_boxes(img_path, coco_boxes, coco_labels, "COCO формат")

        plt.tight_layout()
        plt.show()

    def check_dataset_statistics(self):
        """Перевірка статистики датасету"""
        total_images = 0
        total_objects = 0
        class_counts = {cls: 0 for cls in self.classes}

        for xml_file in self.ann_path.glob("*.xml"):
            total_images += 1
            boxes, labels = self.load_voc_annotation(xml_file)
            total_objects += len(boxes)

            for label in labels:
                class_counts[label] += 1

        print(f"Статистика датасету:")
        print(f"Всього зображень: {total_images}")
        print(f"Всього об'єктів: {total_objects}")
        print("\nРозподіл класів:")
        for cls, count in class_counts.items():
            print(f"{cls}: {count}")

    def validate_conversions(self):
        """Перевірка коректності конвертації"""
        voc_files = list(self.ann_path.glob("*.xml"))
        coco_files = list(self.ann_path.glob("*_cocoformat.json"))

        print(f"Знайдено VOC анотацій: {len(voc_files)}")
        print(f"Знайдено COCO анотацій: {len(coco_files)}")

        for coco_file in coco_files:
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)

            print(f"\nПеревірка {coco_file.name}:")
            print(f"Кількість зображень: {len(coco_data['images'])}")
            print(f"Кількість анотацій: {len(coco_data['annotations'])}")
            print(f"Кількість категорій: {len(coco_data['categories'])}")

            # Перевірка цілісності даних
            image_ids = set(img['id'] for img in coco_data['images'])
            ann_image_ids = set(ann['image_id'] for ann in coco_data['annotations'])

            if not ann_image_ids.issubset(image_ids):
                print("ПОМИЛКА: Знайдено анотації для неіснуючих зображень!")

            category_ids = set(cat['id'] for cat in coco_data['categories'])
            ann_category_ids = set(ann['category_id'] for ann in coco_data['annotations'])

            if not ann_category_ids.issubset(category_ids):
                print("ПОМИЛКА: Знайдено анотації з неіснуючими категоріями!")


def main():
    visualizer = DatasetVisualizer()

    # Перевіряємо статистику датасету
    visualizer.check_dataset_statistics()

    # Перевіряємо коректність конвертації
    visualizer.validate_conversions()

    # Візуалізуємо кілька прикладів
    # Замініть на реальні назви файлів з вашого датасету
    example_images = ["000005", "000012", "000019"]
    for img_name in example_images:
        try:
            visualizer.compare_annotations(img_name)
        except Exception as e:
            print(f"Помилка при візуалізації {img_name}: {e}")


if __name__ == "__main__":
    main()