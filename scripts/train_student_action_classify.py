import comet_ml
from ultralytics import YOLO
from ultralytics.data import augment
from ultralytics.data.augment import ClassifyLetterBox, ToTensor, DEFAULT_MEAN, DEFAULT_STD, classify_augmentations
from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.data.dataset import ClassificationDataset
from typing import Any
import torchvision.transforms as T
import torch
import numpy as np


from ultralytics.data.augment import ClassifyLetterBox, ToTensor, DEFAULT_MEAN, DEFAULT_STD

from PIL import Image, ImageOps

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="student-action-classify")
class SquarePadToMaxSide:

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        pad_w = s - w
        pad_h = s - h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return ImageOps.expand(img, border=padding, fill=(114, 114, 114))


class Dataset(ClassificationDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.torch_transforms = T.Compose([SquarePadToMaxSide(), self.torch_transforms])


class Trainer(ClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return Dataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

model = YOLO("yolo11s-cls.pt")

# Train the model
results = model.train(trainer=Trainer,
                      data=r"/DATA/yujiannan/Datasets/20260210",
                      batch=16*6,
                      epochs=300,
                      imgsz=224,
                      exist_ok=True,
                      project="logs/student_action_classify",
                      name="6")