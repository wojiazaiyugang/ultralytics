from typing import Any

import comet_ml
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="student-action-classify")

def pad_to_square(img: Image.Image, fill=114):
    w, h = img.size
    max_side = max(w, h)
    pad_w = max_side - w
    pad_h = max_side - h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return F.pad(img, padding, fill=fill)

class Dataset(ClassificationDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, i: int) -> dict:
        """
        Return subset of data and targets corresponding to given indices.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        im = pad_to_square(im) # <<<<<
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}


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
                      name="7")