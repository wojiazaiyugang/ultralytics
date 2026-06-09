import random

import cv2
import numpy as np
from PIL import Image

from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer


def random_tight_crop(image: np.ndarray) -> np.ndarray:
    """Randomly remove image margins to simulate tighter detector crops."""
    h, w = image.shape[:2]
    if h < 8 or w < 8:
        return image

    max_margin = 0.18
    left = int(w * random.uniform(0.0, max_margin))
    right = int(w * random.uniform(0.0, max_margin))
    top = int(h * random.uniform(0.0, max_margin))
    bottom = int(h * random.uniform(0.0, max_margin))

    x1, x2 = left, w - right
    y1, y2 = top, h - bottom
    if x2 - x1 < max(4, int(w * 0.55)) or y2 - y1 < max(4, int(h * 0.55)):
        return image
    return image[y1:y2, x1:x2]


def random_expand_pad(image: np.ndarray) -> np.ndarray:
    """Randomly pad/expand an image to simulate crops with more context or boundary padding."""
    h, w = image.shape[:2]
    if h < 2 or w < 2:
        return image

    expand = random.uniform(1.08, 1.45)
    target_w = max(w + 1, int(w * expand))
    target_h = max(h + 1, int(h * expand))
    pad_w = target_w - w
    pad_h = target_h - h
    left = random.randint(0, pad_w)
    right = pad_w - left
    top = random.randint(0, pad_h)
    bottom = pad_h - top

    mode = random.choice(("constant", "replicate", "reflect"))
    if mode == "constant":
        return cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
    if mode == "replicate":
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def random_crop_style(image: np.ndarray) -> np.ndarray:
    """
    Randomize crop style for all classes.

    This reduces leakage where crop tightness or synthetic padding becomes a proxy for a label.
    """
    r = random.random()
    if r < 0.25:
        return random_tight_crop(image)
    if r < 0.50:
        return random_expand_pad(image)
    return image


class CropStyleClassificationDataset(ClassificationDataset):
    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        super().__init__(root=root, args=args, augment=augment, prefix=prefix)
        self.crop_style_augment = augment

    def __getitem__(self, i: int) -> dict:
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:
            im = cv2.imread(f)  # BGR
        if im is None:
            raise FileNotFoundError(f"无法读取图片: {f}")

        if self.crop_style_augment:
            im = random_crop_style(im)

        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}


class CropStyleClassificationTrainer(ClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        if mode == "train":
            return CropStyleClassificationDataset(root=img_path, args=self.args, augment=True, prefix=mode)
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=mode)
