import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import pygame
import time
from threading import Thread
import json
import os
import warnings
import logging
from datetime import datetime
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil
import yaml
from ultralytics import YOLO
import torch
import glob

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

COLORS = {
    "bg_primary": "#f0f0f0",
    "bg_secondary": "#ffffff",
    "text_primary": "#2c3e50",
    "text_secondary": "#34495e",
    "accent": "#001a2b",
    "success": "#27ae60",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "border": "#bdc3c7"
}

class CustomObjectTrainer:
    def __init__(self):
        self.custom_classes = []
        self.custom_model = None
        self.training_in_progress = False
        self.training_progress = None
        self.training_status = None
        self.class_name_mapping = {}
        self.reset_training_state()
        self.previous_model_path = None

    def set_tk_vars(self, progress_var, status_var):
        self.training_progress = progress_var
        self.training_status = status_var
        
    def prepare_training_data(self, images_dir, annotations_dir, output_dir, incremental=False):
        try:
            if not incremental:
                self.reset_training_state()
            
            if not os.path.exists(images_dir):
                raise ValueError(f"Директория с изображениями не существует: {images_dir}")
            if not os.path.exists(annotations_dir):
                raise ValueError(f"Директория с аннотациями не существует: {annotations_dir}")
            
            os.makedirs(output_dir, exist_ok=True)
            for folder in ['train', 'val']:
                os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
                os.makedirs(os.path.join(output_dir, folder, 'images'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, folder, 'labels'), exist_ok=True)
            
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                raise ValueError("Не найдено изображений в указанной директории")
            
            print(f"Найдено {len(image_files)} изображений")
            
            if incremental and os.path.exists(os.path.join(output_dir, 'classes.names')):
                with open(os.path.join(output_dir, 'classes.names'), 'r', encoding='utf-8') as f:
                    existing_classes = [line.strip() for line in f.readlines() if line.strip()]
                self.custom_classes = existing_classes
                print(f"Загружены существующие классы: {self.custom_classes}")
            
            train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42, shuffle=True)
            
            train_count = self._copy_files(train_files, images_dir, annotations_dir, output_dir, 'train', incremental)
            val_count = self._copy_files(val_files, images_dir, annotations_dir, output_dir, 'val', incremental)
            
            if train_count == 0 and val_count == 0:
                raise ValueError("Не удалось обработать ни одного файла с аннотациями")
            
            classes_file_path = os.path.join(output_dir, 'classes.names')
            with open(classes_file_path, 'w', encoding='utf-8') as f:
                for class_name in self.custom_classes:
                    f.write(f"{class_name}\n")
            print(f"Создан файл classes.names с {len(self.custom_classes)} классами")
            
            self._create_yaml_file(output_dir)
            
            print(f"Подготовлено данных: {train_count} train, {val_count} val")
            print(f"Классы: {self.custom_classes}")
            
            return True
        
        except Exception as e:
            print(f"Ошибка подготовки данных: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def _create_yaml_file(self, output_dir):
        try:
            if not self.custom_classes:
                raise ValueError("Нет классов для обучения! Проверьте аннотации.")
            
            names_dict = {}
            for i, class_name in enumerate(self.custom_classes):
                names_dict[i] = class_name
            
            yaml_content = {
                'path': os.path.abspath(output_dir),
                'train': 'train/images',
                'val': 'val/images',
                'nc': len(self.custom_classes),
                'names': names_dict
            }
            
            yaml_path = os.path.join(output_dir, 'data.yaml')
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)
            
            print(f"Создан YAML файл: {yaml_path}")
            print(f"Количество классов: {len(self.custom_classes)}")
            print(f"Имена классов: {list(names_dict.values())}")
            
            return True
            
        except Exception as e:
            print(f"Ошибка создания YAML файла: {e}")
            raise

    def _copy_files(self, files, images_dir, annotations_dir, output_dir, split_type, incremental=False):
        print(f"Начало обработки {len(files)} файлов")
        print(f"Текущие классы перед обработкой: {self.custom_classes}")
        
        processed_count = 0
        
        for file in files:
            try:
                img_dst = os.path.join(output_dir, split_type, 'images', file)
                if incremental and os.path.exists(img_dst):
                    print(f"Файл {file} уже существует, пропускаем")
                    processed_count += 1
                    continue
                
                img_src = os.path.join(images_dir, file)
                shutil.copy2(img_src, img_dst)
                
                base_name = os.path.splitext(file)[0]
                annotation_files = [
                    base_name + '.xml',
                    base_name + '.txt',
                    base_name + '.json'
                ]
                
                annotation_processed = False
                for ann_file in annotation_files:
                    annotation_src = os.path.join(annotations_dir, ann_file)
                    
                    if os.path.exists(annotation_src):
                        if ann_file.endswith('.xml'):
                            temp_set = set(self.custom_classes)
                            self._extract_classes_from_xml(annotation_src, temp_set)
                            self.custom_classes = sorted(list(temp_set))
                            
                            if self._convert_xml_to_yolo(annotation_src, output_dir, split_type, file):
                                annotation_processed = True
                                break
                        elif ann_file.endswith('.txt'):
                            ann_dst = os.path.join(output_dir, split_type, 'labels', base_name + '.txt')
                            shutil.copy2(annotation_src, ann_dst)
                            annotation_processed = True
                            break
                
                if annotation_processed:
                    processed_count += 1
                else:
                    print(f"Предупреждение: не найдены аннотации для {file}")
                    
            except Exception as e:
                print(f"Ошибка обработки файла {file}: {e}")
        
        print(f"После обработки файлов классы: {self.custom_classes}")
        print(f"Обработано файлов: {processed_count}")
        return processed_count
    
    def _extract_classes_from_xml(self, xml_file, classes_set):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is not None and name_elem.text.strip():
                    classes_set.add(name_elem.text.strip())
                    
        except Exception as e:
            print(f"Ошибка чтения XML файла {xml_file}: {e}")

    def _convert_xml_to_yolo(self, xml_file, output_dir, split_type, image_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find('size')
            if size is None:
                print(f"Предупреждение: нет информации о размере в {xml_file}")
                return False
                
            width_elem = size.find('width')
            height_elem = size.find('height')
            
            if width_elem is None or height_elem is None:
                print(f"Предупреждение: неполная информация о размере в {xml_file}")
                return False
                
            width = int(width_elem.text)
            height = int(height_elem.text)
            
            if width == 0 or height == 0:
                print(f"Предупреждение: некорректные размеры в {xml_file}")
                return False
            
            base_name = os.path.splitext(image_file)[0]
            yolo_path = os.path.join(output_dir, split_type, 'labels', base_name + '.txt')
            
            with open(yolo_path, 'w', encoding='utf-8') as f:
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is None:
                        continue
                        
                    class_name = name_elem.text.strip()
                    if not class_name:
                        continue
                    
                    if class_name in self.custom_classes:
                        class_id = self.custom_classes.index(class_name)
                    else:
                        self.custom_classes.append(class_name)
                        class_id = self.custom_classes.index(class_name)
                        print(f"Добавлен новый класс: {class_name}, ID: {class_id}")
                    
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        continue
                        
                    xmin_elem = bbox.find('xmin')
                    ymin_elem = bbox.find('ymin')
                    xmax_elem = bbox.find('xmax')
                    ymax_elem = bbox.find('ymax')
                    
                    if None in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]:
                        continue
                    
                    xmin = float(xmin_elem.text)
                    ymin = float(ymin_elem.text)
                    xmax = float(xmax_elem.text)
                    ymax = float(ymax_elem.text)
                    
                    x_center = (xmin + xmax) / 2 / width
                    y_center = (ymin + ymax) / 2 / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height
                    
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                    else:
                        print(f"Предупреждение: некорректные координаты в {xml_file}")
            
            return True
                        
        except Exception as e:
            print(f"Ошибка конвертации {xml_file}: {e}")
            return False
    
    def find_latest_checkpoint(self):
        try:
            checkpoint_patterns = [
                'runs/detect/train*/weights/last.pt',
                'custom_model_training/*/weights/last.pt',
            ]
            
            latest_checkpoint = None
            latest_time = 0
            
            for pattern in checkpoint_patterns:
                for checkpoint_path in glob.glob(pattern):
                    if os.path.exists(checkpoint_path) and "last.pt" in checkpoint_path:
                        mod_time = os.path.getmtime(checkpoint_path)
                        if mod_time > latest_time:
                            latest_time = mod_time
                            latest_checkpoint = checkpoint_path
            
            return latest_checkpoint
        except Exception as e:
            print(f"Ошибка поиска чекпоинта: {e}")
            return None
        
    def train_custom_model(self, data_dir, epochs=50, img_size=640, batch_size=8, 
                      progress_callback=None, incremental=False, resume=False):
        try:
            if not incremental and not resume:
                self.reset_training_state()
            
            self.training_in_progress = True
            if self.training_status:
                self.training_status.set("Подготовка к обучению...")
            
            data_yaml_path = os.path.join(data_dir, 'data.yaml')
            if not os.path.exists(data_yaml_path):
                raise FileNotFoundError(f"Файл data.yaml не найден в {data_dir}")
            
            train_images_dir = os.path.join(data_dir, 'train', 'images')
            val_images_dir = os.path.join(data_dir, 'val', 'images')
            
            if not os.path.exists(train_images_dir) or len(os.listdir(train_images_dir)) == 0:
                raise ValueError("Нет данных для обучения в train директории")
            if not os.path.exists(val_images_dir) or len(os.listdir(val_images_dir)) == 0:
                raise ValueError("Нет данных для валидации в val директории")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Используемое устройство: {device}")
            
            model_path = None
            
            if incremental and os.path.exists('custom_model.pt'):
                model_path = 'custom_model.pt'
                print(f"Инкрементальное обучение: загружаем модель {model_path}")
            elif resume:
                latest_checkpoint = self.find_latest_checkpoint()
                if latest_checkpoint and os.path.exists(latest_checkpoint):
                    model_path = latest_checkpoint
                    print(f"Продолжение обучения: загружаем чекпоинт {model_path}")
                else:
                    print("Чекпоинт не найден, начинаем новое обучение")
                    model_path = 'yolov8n.pt'
            else:
                model_path = 'yolov8n.pt'
                print("Обучение с нуля: используем базовую модель yolov8n.pt")
            
            if self.training_status:
                self.training_status.set("Загрузка модели...")
            if progress_callback:
                progress_callback(5, "Загрузка модели YOLO")
            
            try:
                if model_path and os.path.exists(model_path):
                    model = YOLO(model_path)
                    print(f"Модель успешно загружена из: {model_path}")
                else:
                    print("Создаем новую модель YOLOv8")
                    model = YOLO('yolov8n.yaml')
            except Exception as e:
                print(f"Ошибка загрузки модели {model_path}: {e}")
                print("Создаем новую модель YOLOv8")
                model = YOLO('yolov8n.yaml')
            
            train_args = {
                'data': data_yaml_path,
                'epochs': epochs,
                'imgsz': img_size,
                'batch': batch_size,
                'device': device,
                'workers': 1,
                'optimizer': 'auto',
                'verbose': True,
                'save': True,
                'exist_ok': True,
                'project': 'custom_model_training',
                'name': datetime.now().strftime('%Y%m%d_%H%M%S'),
            }
            
            if resume and model_path and "last.pt" in model_path:
                train_args['resume'] = True
                print(f"Продолжение обучения с: {model_path}")
            else:
                train_args['resume'] = False
                print("Начало нового обучения")
            
            if self.training_status:
                self.training_status.set("Начало обучения...")
            if progress_callback:
                progress_callback(10, "Начало обучения модели")
            
            results = model.train(**train_args)
            
            best_model_path = results.save_dir
            best_model_file = os.path.join(best_model_path, 'weights', 'best.pt')
            
            if not os.path.exists(best_model_file):
                possible_paths = [
                    os.path.join(best_model_path, 'best.pt'),
                    os.path.join('runs', 'detect', 'train', 'weights', 'best.pt'),
                    os.path.join('runs', 'detect', 'train2', 'weights', 'best.pt'),
                    'best.pt'
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        best_model_file = path
                        break
                else:
                    raise FileNotFoundError("Не удалось найти обученную модель")
            
            if os.path.exists(best_model_file):
                final_model_path = 'custom_model.pt'
                shutil.copy2(best_model_file, final_model_path)
                
                self.previous_model_path = final_model_path
                
                classes_src = os.path.join(data_dir, 'classes.names')
                if os.path.exists(classes_src):
                    shutil.copy2(classes_src, 'classes.names')
                    print("Файл classes.names скопирован рядом с моделью")
                
                self.custom_model = YOLO(final_model_path)
                
                if self.training_status:
                    self.training_status.set("Обучение завершено!")
                if progress_callback:
                    progress_callback(100, "Обучение успешно завершено")
                
                print("Проверка созданных файлов:")
                if os.path.exists(final_model_path):
                    print(f"✓ Модель сохранена: {final_model_path}")
                else:
                    print("✗ Модель не найдена!")
                
                if os.path.exists('classes.names'):
                    with open('classes.names', 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"✓ Содержимое classes.names:\n{content}")
                else:
                    print("✗ Файл classes.names не найден")
                
                return True
            else:
                raise FileNotFoundError("Не удалось найти обученную модель")
                
        except Exception as e:
            error_msg = f"Ошибка обучения: {str(e)}"
            print(error_msg)
            if self.training_status:
                self.training_status.set(f"Ошибка: {str(e)}")
            if progress_callback:
                progress_callback(0, error_msg)
            return False
        finally:
            self.training_in_progress = False
        
    def verify_training_data(self, data_dir):
        try:
            data_yaml_path = os.path.join(data_dir, 'data.yaml')
            if not os.path.exists(data_yaml_path):
                return False, "Файл data.yaml не найден"
            
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            classes_file = os.path.join(data_dir, 'classes.names')
            if not os.path.exists(classes_file):
                return False, "Файл classes.names не найден"
            
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(classes) != yaml_data.get('nc', 0):
                return False, f"Несоответствие: {len(classes)} классов в names vs {yaml_data.get('nc', 0)} в YAML"
            
            for split in ['train', 'val']:
                images_dir = os.path.join(data_dir, split, 'images')
                labels_dir = os.path.join(data_dir, split, 'labels')
                
                if not os.path.exists(images_dir) or not os.listdir(images_dir):
                    return False, f"Нет изображений в {images_dir}"
                
                if not os.path.exists(labels_dir) or not os.listdir(labels_dir):
                    return False, f"Нет аннотаций в {labels_dir}"
            
            return True, f"Данные корректны: {len(classes)} классов"
            
        except Exception as e:
            return False, f"Ошибка проверки данных: {str(e)}"

    def load_custom_model(self, model_path='custom_model.pt'):
        try:
            if os.path.exists(model_path):
                if hasattr(self, 'custom_model') and self.custom_model is not None:
                    try:
                        del self.custom_model
                    except:
                        pass
                    self.custom_model = None
                
                self.custom_model = YOLO(model_path)
                
                if hasattr(self.custom_model, 'names') and self.custom_model.names:
                    self.custom_classes = list(self.custom_model.names.values())
                    print(f"Кастомная модель загружена! Классы из модели: {self.custom_classes}")
                else:
                    model_dir = os.path.dirname(model_path)
                    classes_files = [
                        os.path.join(model_dir, 'classes.names'),
                        'classes.names',
                        os.path.join('custom_data', 'classes.names')
                    ]
                    
                    classes_loaded = False
                    for classes_file in classes_files:
                        if os.path.exists(classes_file):
                            with open(classes_file, 'r', encoding='utf-8') as f:
                                self.custom_classes = [line.strip() for line in f.readlines() if line.strip()]
                            print(f"Классы загружены из файла {classes_file}: {self.custom_classes}")
                            classes_loaded = True
                            break
                    
                    if not classes_loaded:
                        print("Предупреждение: не удалось найти файл classes.names")
                        self.custom_classes = []
                
                return True
        except Exception as e:
            print(f"Ошибка загрузки кастомной модели: {e}")
            import traceback
            traceback.print_exc()
        return False
    
    def detect_with_custom_model(self, frame, conf_threshold=0.5):
        if self.custom_model is None:
            print("Кастомная модель не загружена!")
            return frame, False, []
            
        print(f"Детекция с классами: {self.custom_classes}")
        
        try:
            results = self.custom_model(frame, conf=conf_threshold, verbose=False)
            
            detections = []
            danger_detected = False
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    print(f"Найдено объектов: {len(class_ids)}")
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = map(int, box)
                        
                        label = "unknown"
                        if hasattr(self.custom_model, 'names') and self.custom_model.names is not None:
                            if class_id in self.custom_model.names:
                                label = self.custom_model.names[class_id]
                            else:
                                label = f"class_{class_id}"
                        elif class_id < len(self.custom_classes):
                            label = self.custom_classes[class_id]
                        else:
                            label = f"class_{class_id}"
                        
                        print(f"Объект {i}: class_id={class_id}, label={label}, conf={conf:.2f}")
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'label': label
                        })
            
            return frame, danger_detected, detections
            
        except Exception as e:
            print(f"Ошибка детекции кастомной моделью: {e}")
            return frame, False, []
        
    def reset_training_state(self):
        self.custom_classes = []
        self.class_name_mapping = {}
        if hasattr(self, 'custom_model') and self.custom_model is not None:
            try:
                del self.custom_model
            except:
                pass
            self.custom_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class ImageAnnotator:
    def __init__(self):
        self.current_image = None
        self.image_path = None
        self.annotations = []
        self.current_class = "pointing_finger"
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.classes = ["pointing_finger", "thumbs_up", "peace_sign", "fist", "palm"]
        
    def load_image(self, image_path):
        self.image_path = image_path
        self.current_image = cv2.imread(image_path)
        self.annotations = []
        return self.current_image.copy()
    
    def start_drawing(self, x, y):
        self.drawing = True
        self.start_x, self.start_y = x, y
        
    def draw_bbox(self, x, y, image):
        if self.drawing:
            temp_image = image.copy()
            cv2.rectangle(temp_image, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
            return temp_image
        return image
    
    def end_drawing(self, x, y, class_name):
        if self.drawing:
            x1, y1 = min(self.start_x, x), min(self.start_y, y)
            x2, y2 = max(self.start_x, x), max(self.start_y, y)
            
            if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
                self.drawing = False
                return False
            
            if class_name and class_name not in self.classes:
                self.classes.append(class_name)
            
            self.annotations.append({
                'class': class_name,
                'bbox': [x1, y1, x2, y2]
            })
            self.drawing = False
            return True
        return False
    
    def save_annotations(self, output_dir):
        if not self.image_path or not self.annotations:
            return False
        
        try:
            root = ET.Element("annotation")
            
            filename_elem = ET.SubElement(root, "filename")
            filename_elem.text = os.path.basename(self.image_path)
            
            size_elem = ET.SubElement(root, "size")
            height, width = self.current_image.shape[:2]
            
            width_elem = ET.SubElement(size_elem, "width")
            width_elem.text = str(width)
            height_elem = ET.SubElement(size_elem, "height")
            height_elem.text = str(height)
            depth_elem = ET.SubElement(size_elem, "depth")
            depth_elem.text = "3"
            
            for ann in self.annotations:
                obj_elem = ET.SubElement(root, "object")
                
                name_elem = ET.SubElement(obj_elem, "name")
                name_elem.text = ann['class']
                
                bndbox_elem = ET.SubElement(obj_elem, "bndbox")
                xmin_elem = ET.SubElement(bndbox_elem, "xmin")
                xmin_elem.text = str(ann['bbox'][0])
                ymin_elem = ET.SubElement(bndbox_elem, "ymin")
                ymin_elem.text = str(ann['bbox'][1])
                xmax_elem = ET.SubElement(bndbox_elem, "xmax")
                xmax_elem.text = str(ann['bbox'][2])
                ymax_elem = ET.SubElement(bndbox_elem, "ymax")
                ymax_elem.text = str(ann['bbox'][3])
            
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            xml_path = os.path.join(output_dir, f"{base_name}.xml")
            
            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения аннотаций: {e}")
            return False
        
class SafetyMonitoringSystem:
    def __init__(self, config_file="config.json"):
        pygame.mixer.init()
        self.trainer = CustomObjectTrainer()
        self.load_config(config_file)
        self.init_models()
        self.running = False
        self.alert_active = False
        self.video_source = self.config["video_source"]
        self.logger = None
        self.setup_logger()
    
    def setup_logger(self):
        self.logger = logging.getLogger("SafetyMonitoringSystem")
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        self.log_file_handler = logging.FileHandler("safety_log.txt", encoding='utf-8')
        self.log_file_handler.setFormatter(formatter)
        
        self.logging_enabled = False
    
    def start_logging(self):
        if not self.logging_enabled:
            self.logger.addHandler(self.log_file_handler)
            self.logging_enabled = True
            self.logger.info("Логирование активировано")
    
    def stop_logging(self):
        if self.logging_enabled:
            self.logger.info("Логирование деактивировано")
            self.logger.removeHandler(self.log_file_handler)
            self.logging_enabled = False
    
    def log_event(self, message, level="info"):
        if self.logging_enabled:
            try:
                if level.lower() == "info":
                    self.logger.info(message)
                elif level.lower() == "warning":
                    self.logger.warning(message)
                elif level.lower() == "error":
                    self.logger.error(message)
            except UnicodeEncodeError:
                safe_message = message.encode('ascii', errors='replace').decode('ascii')
                self.logger.warning(f"Encoding error, safe message: {safe_message}")

    def load_config(self, config_file):
        default_config = {
            "video_source": 0,
            "alert_sound": "alert.wav",
            "visual_alert_color": [0, 0, 255],
            "danger_zones": [],
            "danger_objects": ["knife", "gun", "fire", "machine", "cell phone"],
            "motion_detection": True,
            "thermal_detection": False,
            "alert_types": ["visual", "audio"],
            "min_confidence": 0.7
        }
    
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config = {**default_config, **user_config}
        except (FileNotFoundError, json.JSONDecodeError):
            self.config = default_config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
    
        if "audio" in self.config["alert_types"]:
            try:
                self.alert_sound = pygame.mixer.Sound(self.config["alert_sound"])
            except Exception as e:
                print(f"Ошибка загрузки звука: {e}")
                self.config["alert_types"].remove("audio")

    def init_models(self):
        try:
            if os.path.exists('custom_model.pt'):
                self.trainer.reset_training_state()
                if self.trainer.load_custom_model('custom_model.pt'):
                    print("Кастомная модель загружена!")
                    print(f"Классы кастомной модели: {self.trainer.custom_classes}")
                else:
                    print("Ошибка загрузки кастомной модели")
            else:
                print("Кастомная модель не найдена")
        
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
        
    def start(self, video_source=None):
        self.running = True
        source = video_source if video_source is not None else self.video_source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            print(f"Не удалось открыть источник {source}, пробую камеру 0")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Камера не найдена, использую тестовое видео")
                self.cap = cv2.VideoCapture("test_video.mp4")
        
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.video_thread = Thread(target=self.process_video)
        self.video_thread.start()
        self.alert_thread = Thread(target=self.alert_monitor)
        self.alert_thread.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'video_thread'):
            self.video_thread.join()
        if hasattr(self, 'alert_thread'):
            self.alert_thread.join()
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
    
    def process_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame, danger_detected = self.detect_dangers(frame)
            cv2.imshow("Safety Monitoring", processed_frame)
            self.alert_active = danger_detected
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
    
    def detect_dangers(self, frame):
        danger_detected = False
        
        frame, custom_danger, custom_detections = self.trainer.detect_with_custom_model(frame, self.config["min_confidence"])
        
        if custom_detections:
            danger_color = tuple(map(int, self.config["visual_alert_color"][::-1]))
            safe_color = (255, 0, 0)
            font = cv2.FONT_HERSHEY_PLAIN
            
            for detection in custom_detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                label = detection['label']
                
                danger_objects = [obj.strip().lower() for obj in self.config["danger_objects"]]
                
                is_dangerous = any(danger_obj in label.lower() for danger_obj in danger_objects)
                
                if not is_dangerous and label.startswith('class_'):
                    try:
                        class_idx = int(label.split('_')[1])
                        if class_idx < len(danger_objects) and danger_objects[class_idx]:
                            is_dangerous = True
                    except (IndexError, ValueError):
                        pass
                
                if is_dangerous:
                    danger_detected = True
                    color = danger_color
                    display_label = f"DANGER: {label}"
                    self.log_event(f"Обнаружен кастомный объект: {label}", "warning")
                else:
                    color = safe_color
                    display_label = f"Custom: {label}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{display_label} {confidence:.2f}", 
                           (x1, y1 - 5), font, 1, color, 1)
        
        danger_detected = danger_detected or custom_danger
        return frame, danger_detected
    
    def alert_monitor(self):
        while self.running:
            if self.alert_active:
                if "audio" in self.config["alert_types"]:
                    self.alert_sound.play()
                    self.log_event("Активировано звуковое оповещение")
                time.sleep(2)
            else:
                time.sleep(0.1)
        
    def save_config(self):
        with open("config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

class TrainingWindow(tk.Toplevel):
    def __init__(self, parent, system):
        super().__init__(parent)
        self.system = system
        self.title("Обучение кастомной модели YOLOv8")
        self.geometry("700x500")
        self.iconbitmap("security_icon.ico")
        self.configure(bg=COLORS["bg_primary"])
        self.create_widgets()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        training_type_frame = ttk.LabelFrame(main_frame, text="Тип обучения", padding="10")
        training_type_frame.pack(fill=tk.X, pady=5)
        
        self.training_type_var = tk.StringVar(value="new")
        ttk.Radiobutton(training_type_frame, text="Новое обучение", 
                       variable=self.training_type_var, value="new").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(training_type_frame, text="Инкрементальное обучение", 
                       variable=self.training_type_var, value="incremental").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(training_type_frame, text="Продолжение обучения", 
                       variable=self.training_type_var, value="resume").pack(anchor=tk.W, pady=2)
        
        ttk.Label(main_frame, textvariable=self.system.trainer.training_status, 
                 font=('Helvetica', 10), foreground=COLORS["text_secondary"]).pack(pady=5)
        ttk.Progressbar(main_frame, variable=self.system.trainer.training_progress, 
                       length=600).pack(pady=10)
        
        params_frame = ttk.LabelFrame(main_frame, text="Параметры обучения", padding="10")
        params_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(params_frame, text="Эпохи:", foreground=COLORS["text_primary"]).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(params_frame, from_=10, to=500, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="Размер изображения:", foreground=COLORS["text_primary"]).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.img_size_var = tk.IntVar(value=640)
        ttk.Combobox(params_frame, textvariable=self.img_size_var, 
                    values=[320, 416, 640, 1280], width=8).grid(row=0, column=3, padx=5)
        
        ttk.Label(params_frame, text="Размер батча:", foreground=COLORS["text_primary"]).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.batch_var = tk.IntVar(value=8)
        ttk.Spinbox(params_frame, from_=1, to=64, textvariable=self.batch_var, width=10).grid(row=1, column=1, padx=5)
        
        dirs_frame = ttk.LabelFrame(main_frame, text="Директории", padding="10")
        dirs_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(dirs_frame, text="Изображения:", foreground=COLORS["text_primary"]).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.images_var = tk.StringVar()
        ttk.Entry(dirs_frame, textvariable=self.images_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(dirs_frame, text="Обзор", 
                  command=lambda: self.browse_directory(self.images_var)).grid(row=0, column=2, padx=5)
        
        ttk.Label(dirs_frame, text="Аннотации:", foreground=COLORS["text_primary"]).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.annotations_var = tk.StringVar()
        ttk.Entry(dirs_frame, textvariable=self.annotations_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(dirs_frame, text="Обзор", 
                  command=lambda: self.browse_directory(self.annotations_var)).grid(row=1, column=2, padx=5)
        
        ttk.Label(dirs_frame, text="Выходная директория:", foreground=COLORS["text_primary"]).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_var = tk.StringVar(value="custom_data")
        ttk.Entry(dirs_frame, textvariable=self.output_var, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(dirs_frame, text="Обзор", 
                  command=lambda: self.browse_directory(self.output_var)).grid(row=2, column=2, padx=5)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        self.prepare_btn = ttk.Button(btn_frame, text="Подготовить данные", 
                                     command=self.prepare_data)
        self.prepare_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(btn_frame, text="Обучить модель", 
                                   command=self.train_model, state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Закрыть", 
                  command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        self.training_type_var.trace_add("write", self.on_training_type_changed)
        self.on_training_type_changed()
        
        self.data_prepared = False
        
    def on_training_type_changed(self, *args):
        training_type = self.training_type_var.get()
        
        if training_type == "incremental":
            if not os.path.exists('custom_model.pt'):
                messagebox.showwarning("Предупреждение", 
                                     "Для инкрементального обучения требуется существующая модель. "
                                     "Сначала обучите модель с нуля.")
                self.training_type_var.set("new")
                self.train_btn.config(state=tk.DISABLED)
            else:
                if self.data_prepared:
                    self.train_btn.config(state=tk.NORMAL)
                else:
                    self.train_btn.config(state=tk.DISABLED)
                self.system.trainer.training_status.set("Готов к инкрементальному обучению")
        
        elif training_type == "resume":
            latest_checkpoint = self.system.trainer.find_latest_checkpoint()
            if not latest_checkpoint:
                messagebox.showwarning("Предупреждение", 
                                     "Для продолжения обучения требуется чекпоинт. "
                                     "Сначала обучите модель.")
                self.training_type_var.set("new")
                self.train_btn.config(state=tk.DISABLED)
            else:
                if self.data_prepared:
                    self.train_btn.config(state=tk.NORMAL)
                else:
                    self.train_btn.config(state=tk.DISABLED)
                self.system.trainer.training_status.set("Готов к продолжению обучения")
        
        else: 
            if self.data_prepared:
                self.train_btn.config(state=tk.NORMAL)
            else:
                self.train_btn.config(state=tk.DISABLED)
            self.system.trainer.training_status.set("Готов к обучению кастомной модели")
    
    def browse_directory(self, var):
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
    
    def prepare_data(self):
        try:
            incremental = self.training_type_var.get() == "incremental"
            
            if not incremental:
                self.system.trainer.reset_training_state()
            
            self.prepare_btn.config(state=tk.DISABLED)
            self.system.trainer.training_status.set("Подготовка данных...")
            
            success = self.system.trainer.prepare_training_data(
                self.images_var.get(),
                self.annotations_var.get(),
                self.output_var.get(),
                incremental=incremental
            )
            
            if success:
                messagebox.showinfo("Успех", "Данные успешно подготовлены!")
                self.data_prepared = True
                self.train_btn.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Ошибка", "Не удалось подготовить данные")
                self.data_prepared = False
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка подготовки данных: {str(e)}")
            self.data_prepared = False
        finally:
            self.prepare_btn.config(state=tk.NORMAL)
            self.on_training_type_changed()
    
    def train_model(self):
        def training_thread():
            try:
                training_type = self.training_type_var.get()
                incremental = training_type == "incremental"
                resume = training_type == "resume"
                
                if incremental and not os.path.exists('custom_model.pt'):
                    messagebox.showerror("Ошибка", 
                                       "Для инкрементального обучения требуется существующая модель 'custom_model.pt'")
                    return
                
                success, message = self.system.trainer.verify_training_data(self.output_var.get())
                if not success:
                    messagebox.showerror("Ошибка данных", message)
                    return
                
                print(f"Проверка данных: {message}")
                
                data_yaml_path = os.path.join(self.output_var.get(), 'data.yaml')
                if not os.path.exists(data_yaml_path):
                    messagebox.showerror("Ошибка", "Сначала подготовьте данные!")
                    return
                    
                self.train_btn.config(state=tk.DISABLED)
                
                if not incremental and not resume:
                    self.system.trainer.reset_training_state()
                
                success = self.system.trainer.train_custom_model(
                    self.output_var.get(),
                    epochs=self.epochs_var.get(),
                    img_size=self.img_size_var.get(),
                    batch_size=self.batch_var.get(),
                    progress_callback=self.update_progress,
                    incremental=incremental,
                    resume=resume
                )
                
                if success:
                    messagebox.showinfo("Успех", "Модель успешно обучена!")
                    self.system.init_models()
                else:
                    messagebox.showerror("Ошибка", "Не удалось обучить модель")
                    
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка обучения: {str(e)}")
            finally:
                self.train_btn.config(state=tk.NORMAL)
                self.data_prepared = False
                self.on_training_type_changed()
        
        Thread(target=training_thread, daemon=True).start()
    
    def update_progress(self, progress, status):
        self.system.trainer.training_progress.set(progress)
        self.system.trainer.training_status.set(status)

class AnnotationWindow(tk.Toplevel):
    def __init__(self, parent, system):
        super().__init__(parent)
        self.system = system
        self.annotator = ImageAnnotator()
        self.title("Разметка изображений")
        self.geometry("900x700")
        self.iconbitmap("security_icon.ico")
        self.configure(bg=COLORS["bg_primary"])
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.create_widgets()
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Загрузить изображения", 
                  command=self.load_images).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Предыдущее", 
                  command=self.previous_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Следующее", 
                  command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        class_frame = ttk.Frame(control_frame)
        class_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(class_frame, text="Класс:", foreground=COLORS["text_primary"]).pack(side=tk.LEFT)
        
        self.class_var = tk.StringVar(value="pointing_finger")
        self.class_combo = ttk.Combobox(class_frame, textvariable=self.class_var, 
                                      values=self.annotator.classes, width=20)
        self.class_combo.pack(side=tk.LEFT, padx=5)
        self.class_combo.bind("<KeyRelease>", self.on_class_combo_keyrelease)
        
        ttk.Button(class_frame, text="+", width=3,
                  command=self.add_new_class).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(class_frame, text="Управление классами", 
                  command=self.manage_classes).pack(side=tk.LEFT, padx=5)
        
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(expand=True, fill=tk.BOTH, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg="black", cursor="cross")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(info_frame, text="Инструкция:", foreground=COLORS["text_primary"]).pack(side=tk.LEFT)
        ttk.Label(info_frame, text="1. Зажмите ЛКМ и выделите объект 2. Отпустите для создания bounding box", 
                 foreground=COLORS["text_secondary"]).pack(side=tk.LEFT, padx=5)
        
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(bottom_frame, text="Сохранить аннотации", 
                  command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="Удалить последнюю", 
                  command=self.delete_last_annotation).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="Очистить все", 
                  command=self.clear_annotations).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="Авторазметка", 
                  command=self.auto_annotate).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="Закрыть", 
                  command=self.destroy).pack(side=tk.RIGHT, padx=5)
        
        self.status_var = tk.StringVar(value="Загрузите изображения")
        ttk.Label(bottom_frame, textvariable=self.status_var, foreground=COLORS["text_secondary"]).pack(side=tk.LEFT, padx=5)
        
        self.image_files = []
        self.current_image_index = -1
        self.original_image = None
        
    def on_class_combo_keyrelease(self, event):
        current_text = self.class_var.get()
        if current_text and current_text not in self.annotator.classes:
            self.class_combo['values'] = list(self.annotator.classes) + [current_text]
            
    def add_new_class(self):
        new_class = self.class_var.get().strip()
        if new_class:
            if new_class not in self.annotator.classes:
                self.annotator.classes.append(new_class)
                self.class_combo['values'] = self.annotator.classes
                messagebox.showinfo("Успех", f"Класс '{new_class}' добавлен")
            else:
                messagebox.showinfo("Информация", f"Класс '{new_class}' уже существует")
        else:
            messagebox.showwarning("Предупреждение", "Введите название класса")
            
    def manage_classes(self):
        class_win = tk.Toplevel(self)
        class_win.title("Управление классами")
        class_win.geometry("400x300")
        class_win.configure(bg=COLORS["bg_primary"])
        
        main_frame = ttk.Frame(class_win, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(main_frame, text="Существующие классы:", foreground=COLORS["text_primary"]).pack(pady=5)
        
        class_listbox = tk.Listbox(main_frame, height=10)
        class_listbox.pack(expand=True, fill=tk.BOTH, pady=5)
        
        for cls in self.annotator.classes:
            class_listbox.insert(tk.END, cls)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        def delete_selected():
            selection = class_listbox.curselection()
            if selection:
                class_to_delete = self.annotator.classes[selection[0]]
                if messagebox.askyesno("Подтверждение", f"Удалить класс '{class_to_delete}'?"):
                    del self.annotator.classes[selection[0]]
                    class_listbox.delete(selection[0])
                    self.class_combo['values'] = self.annotator.classes
        
        ttk.Button(btn_frame, text="Удалить выбранный", 
                  command=delete_selected).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Закрыть", 
                  command=class_win.destroy).pack(side=tk.RIGHT, padx=5)
        
    def load_images(self):
        files = filedialog.askopenfilenames(
            title="Выберите изображения для разметки",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if files:
            self.image_files = list(files)
            self.current_image_index = 0
            self.load_current_image()
            
    def load_current_image(self):
        if 0 <= self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {image_path}")
                return
                
            self.annotator.image_path = image_path
            self.annotator.current_image = self.original_image.copy()
            self.annotator.annotations = []
            self.display_image(self.original_image)
            self.status_var.set(f"Изображение {self.current_image_index + 1}/{len(self.image_files)}")
            
    def display_image(self, image):
        self.canvas.delete("all")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        self.orig_height, self.orig_width = height, width
        
        max_size = 800
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            self.scale_x = new_width / width
            self.scale_y = new_height / height
        else:
            new_width, new_height = width, height
            self.scale_x = 1.0
            self.scale_y = 1.0
        
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
        
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        self.draw_existing_annotations()
        
    def draw_existing_annotations(self):
        for ann in self.annotator.annotations:
            x1 = int(ann['bbox'][0] * self.scale_x)
            y1 = int(ann['bbox'][1] * self.scale_y)
            x2 = int(ann['bbox'][2] * self.scale_x)
            y2 = int(ann['bbox'][3] * self.scale_y)
            
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2)
            self.canvas.create_text(x1, y1-15, text=ann['class'], fill="green", 
                                  font=("Arial", 10, "bold"), anchor=tk.NW)
            
    def on_mouse_press(self, event):
        orig_x = int(event.x / self.scale_x)
        orig_y = int(event.y / self.scale_y)
        self.annotator.start_drawing(orig_x, orig_y)
        
    def on_mouse_drag(self, event):
        if self.annotator.drawing:
            temp_image = self.original_image.copy()
            
            orig_x = int(event.x / self.scale_x)
            orig_y = int(event.y / self.scale_y)
            
            cv2.rectangle(temp_image, 
                         (self.annotator.start_x, self.annotator.start_y),
                         (orig_x, orig_y), (0, 255, 0), 2)
            
            temp_image_rgb = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
            if max(self.orig_width, self.orig_height) > 800:
                scale = 800 / max(self.orig_width, self.orig_height)
                new_width = int(self.orig_width * scale)
                new_height = int(self.orig_height * scale)
                temp_image_rgb = cv2.resize(temp_image_rgb, (new_width, new_height))
            
            self.temp_photo = ImageTk.PhotoImage(image=Image.fromarray(temp_image_rgb))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.temp_photo)
            self.draw_existing_annotations()
            
    def on_mouse_release(self, event):
        if self.annotator.drawing:
            orig_x = int(event.x / self.scale_x)
            orig_y = int(event.y / self.scale_y)
            
            class_name = self.class_var.get().strip()
            if not class_name:
                messagebox.showwarning("Предупреждение", "Введите название класса")
                self.annotator.drawing = False
                self.display_image(self.original_image)
                return
                
            if self.annotator.end_drawing(orig_x, orig_y, class_name):
                self.display_image(self.original_image)
            else:
                self.display_image(self.original_image)
                
    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            
    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
            
    def save_annotations(self):
        if not self.annotator.image_path:
            messagebox.showwarning("Предупреждение", "Нет загруженного изображения")
            return
            
        if not self.annotator.annotations:
            messagebox.showwarning("Предупреждение", "Нет аннотаций для сохранения")
            return
            
        output_dir = filedialog.askdirectory(title="Выберите папку для сохранения аннотаций")
        if output_dir:
            try:
                self.annotator.current_image = self.original_image
                if self.annotator.save_annotations(output_dir):
                    messagebox.showinfo("Успех", "Аннотации успешно сохранены!")
                    self.save_class_list(output_dir)
                else:
                    messagebox.showerror("Ошибка", "Не удалось сохранить аннотации")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении: {str(e)}")
                
    def save_class_list(self, output_dir):
        try:
            class_file = os.path.join(output_dir, "classes.txt")
            with open(class_file, 'w', encoding='utf-8') as f:
                for cls in self.annotator.classes:
                    f.write(f"{cls}\n")
        except Exception as e:
            print(f"Ошибка сохранения списка классов: {e}")
                
    def delete_last_annotation(self):
        if self.annotator.annotations:
            self.annotator.annotations.pop()
            self.display_image(self.original_image)
            
    def clear_annotations(self):
        self.annotator.annotations = []
        self.display_image(self.original_image)
        
    def auto_annotate(self):
        if not self.annotator.image_path:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
            
        try:
            image = self.original_image.copy()
            height, width = image.shape[:2]
            
            if not hasattr(self.system, 'net') or not hasattr(self.system, 'output_layers'):
                messagebox.showwarning("Предупреждение", "Модель YOLO не загружена")
                return
                
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.system.net.setInput(blob)
            outs = self.system.net.forward(self.system.output_layers)
            
            conf_threshold = 0.5
            class_ids = []
            confidences = []
            boxes = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > conf_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            added_count = 0
            for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
                if class_id < len(self.system.classes):
                    class_name = self.system.classes[class_id]
                    x, y, w, h = box
                    
                    if w > 10 and h > 10:
                        self.annotator.annotations.append({
                            'class': class_name,
                            'bbox': [x, y, x + w, y + h]
                        })
                        added_count += 1
            
            self.display_image(image)
            messagebox.showinfo("Успех", f"Автоматически размечено {added_count} объектов")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка автоматической разметки: {e}")

class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, system):
        super().__init__(parent)
        self.system = system
        self.title("Настройки конфигурации")
        self.geometry("600x500")
        self.iconbitmap("security_icon.ico")
        self.configure(bg=COLORS["bg_primary"])
        self.create_widgets()
    
    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="Основные")
        self.create_general_tab(general_frame)
        
        objects_frame = ttk.Frame(notebook)
        notebook.add(objects_frame, text="Объекты")
        self.create_objects_tab(objects_frame)
        
        alerts_frame = ttk.Frame(notebook)
        notebook.add(alerts_frame, text="Оповещения")
        self.create_alerts_tab(alerts_frame)
        
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Сохранить", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side=tk.LEFT, padx=5)
    
    def create_general_tab(self, parent):
        ttk.Label(parent, text="Минимальная уверенность обнаружения:", foreground=COLORS["text_primary"]).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.confidence_var = tk.DoubleVar(value=self.system.config["min_confidence"])
        ttk.Scale(parent, from_=0.1, to=1.0, variable=self.confidence_var, 
                 orient=tk.HORIZONTAL, length=300).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(parent, textvariable=self.confidence_var, foreground=COLORS["accent"]).grid(row=0, column=2, padx=5)
        
        ttk.Label(parent, text="Цвет предупреждения (RGB):", foreground=COLORS["text_primary"]).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        color_frame = ttk.Frame(parent)
        color_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        self.color_vars = []
        for i, color in enumerate(["R", "G", "B"]):
            ttk.Label(color_frame, text=color, foreground=COLORS["text_primary"]).grid(row=0, column=i*2, padx=2)
            var = tk.IntVar(value=self.system.config["visual_alert_color"][i])
            self.color_vars.append(var)
            ttk.Spinbox(color_frame, from_=0, to=255, width=3, textvariable=var).grid(row=0, column=i*2+1, padx=2)
        
        self.color_demo = tk.Label(parent, text="Пример", width=15, relief=tk.SUNKEN)
        self.color_demo.grid(row=2, column=1, pady=5, sticky=tk.W)
        self.update_color_demo()
        
        for var in self.color_vars:
            var.trace_add("write", lambda *args: self.update_color_demo())
    
    def create_objects_tab(self, parent):
        ttk.Label(parent, text="Опасные объекты (по одному в строке):", foreground=COLORS["text_primary"]).pack(pady=(5,0))
        
        self.objects_text = scrolledtext.ScrolledText(parent, width=40, height=15, wrap=tk.WORD)
        self.objects_text.pack(padx=5, pady=5, expand=True, fill=tk.BOTH)
        
        objects = "\n".join(self.system.config["danger_objects"])
        self.objects_text.insert(tk.END, objects)
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="Добавить стандартные", 
                  command=self.add_default_objects).pack(side=tk.LEFT, padx=5)
    
    def create_alerts_tab(self, parent):
        ttk.Label(parent, text="Типы оповещений:", foreground=COLORS["text_primary"]).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.alert_vars = {}
        for i, alert_type in enumerate(["visual", "audio"]):
            var = tk.BooleanVar(value=alert_type in self.system.config["alert_types"])
            self.alert_vars[alert_type] = var
            cb = ttk.Checkbutton(parent, text=alert_type.capitalize(), variable=var)
            cb.grid(row=i+1, column=0, sticky=tk.W, padx=20, pady=2)
        
        ttk.Label(parent, text="Дополнительные функции:", foreground=COLORS["text_primary"]).grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.motion_var = tk.BooleanVar(value=self.system.config["motion_detection"])
        ttk.Checkbutton(parent, text="Обнаружение движения", variable=self.motion_var).grid(
            row=4, column=0, sticky=tk.W, padx=20, pady=2)
    
    def update_color_demo(self):
        color = [var.get() for var in self.color_vars]
        hex_color = "#%02x%02x%02x" % tuple(color)
        self.color_demo.config(bg=hex_color, fg="white" if sum(color) < 384 else "black")
    
    def add_default_objects(self):
        default_objects = [
            "knife", "gun", "pistol", "rifle", "handgun", 
            "fire", "flame", "weapon", "cell phone", "explosive", "bomb"
        ]
        self.objects_text.insert(tk.END, "\n" + "\n".join(default_objects))
    
    def save_settings(self):
        try:
            self.system.config["min_confidence"] = self.confidence_var.get()
            self.system.config["visual_alert_color"] = [var.get() for var in self.color_vars]
            
            objects_text = self.objects_text.get("1.0", tk.END).strip()
            self.system.config["danger_objects"] = [
                obj.strip() for obj in objects_text.split("\n") if obj.strip()
            ]
            
            self.system.config["alert_types"] = [
                alert_type for alert_type, var in self.alert_vars.items() if var.get()
            ]
            
            self.system.config["motion_detection"] = self.motion_var.get()
            
            self.system.save_config()
            messagebox.showinfo("Успех", "Настройки успешно сохранены!")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки: {str(e)}")

class Application(tk.Tk):
    def __init__(self, system):
        super().__init__()
        self.system = system
        self.title("Система мониторинга безопасности")
        self.geometry("500x450")
        self.iconbitmap("security_icon.ico")
        self.configure(bg=COLORS["bg_primary"])
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.system.trainer.set_tk_vars(
            tk.DoubleVar(value=0),
            tk.StringVar(value="Готов к обучению")
        )
        
        self.create_widgets()
    
    def create_widgets(self):
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 12), background=COLORS["accent"])
        style.configure('TLabel', font=('Helvetica', 12), background=COLORS["bg_primary"], foreground=COLORS["text_primary"])
        style.configure('TLabelframe', background=COLORS["bg_primary"], foreground=COLORS["text_primary"])
        style.configure('TLabelframe.Label', background=COLORS["bg_primary"], foreground=COLORS["text_primary"])
        
        self.main_frame = ttk.Frame(self, padding="20")
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(self.main_frame, text="Система мониторинга безопасности", 
                 font=('Helvetica', 14, 'bold'), foreground=COLORS["text_primary"]).pack(pady=10)
        
        try:
            img = Image.open("security_icon.png")
            img = img.resize((100, 100), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(img)
            ttk.Label(self.main_frame, image=self.logo).pack(pady=10)
        except:
            pass
        
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Запуск с камеры", 
                  command=self.start_camera).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Запуск с видеофайла", 
                  command=self.start_video_file).pack(side=tk.LEFT, padx=10)
        
        self.status_var = tk.StringVar(value="Готов к работе")
        ttk.Label(self.main_frame, textvariable=self.status_var, 
                 font=('Helvetica', 11), foreground=COLORS["text_secondary"]).pack(pady=10)
        
        self.stop_btn = ttk.Button(self.main_frame, text="Остановить", 
                                  state=tk.DISABLED, command=self.stop)
        self.stop_btn.pack(pady=10)
        
        bottom_frame = ttk.Frame(self.main_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        ttk.Button(bottom_frame, text="Разметка изображений", 
                  command=self.open_annotation).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="Обучение модели", 
                  command=self.open_training).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="Настройки", 
                  command=self.open_settings).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="Отладка", 
                  command=self.debug_detection).pack(side=tk.LEFT, padx=5)
        
        log_frame = ttk.Frame(self.main_frame)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.log_btn = ttk.Button(log_frame, text="Включить логирование", 
                                 command=self.toggle_logging)
        self.log_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(log_frame, text="Выход", 
                  command=self.on_close).pack(side=tk.RIGHT, padx=5)
    
    def open_annotation(self):
        AnnotationWindow(self, self.system)

    def start_camera(self):
        self.status_var.set("Запуск с камеры...")
        self.update()
        self.system.start(0)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Работа с камерой")
    
    def start_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if file_path:
            self.status_var.set(f"Загрузка файла...")
            self.update()
            self.system.start(file_path)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Обработка: {os.path.basename(file_path)}")
    
    def stop(self):
        self.system.stop()
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Система остановлена")
    
    def open_settings(self):
        SettingsWindow(self, self.system)
    
    def open_training(self):
        TrainingWindow(self, self.system)
    
    def on_close(self):
        if messagebox.askokcancel("Выход", "Вы уверены, что хотите выйти?"):
            self.system.stop()
            self.destroy()
    
    def debug_detection(self):
        self.system.show_detection_info()

    def toggle_logging(self):
        if self.system.logging_enabled:
            self.system.stop_logging()
            self.log_btn.config(text="Включить логирование")
            self.status_var.set("Логирование выключено")
        else:
            self.system.start_logging()
            self.log_btn.config(text="Выключить логирование")
            self.status_var.set("Логирование активировано")

if __name__ == "__main__":
    sms = SafetyMonitoringSystem()
    app = Application(sms)
    app.mainloop()