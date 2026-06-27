from copy import copy
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

from crop_style_classification import random_crop_style


def letter_box(image: np.ndarray) -> np.ndarray:
    """
    Pad an image to a centered square without cropping.
    """
    imh, imw = image.shape[:2]
    side = max(imh, imw)
    top = (side - imh) // 2
    left = (side - imw) // 2

    output = np.full((side, side, 3), 114, dtype=image.dtype)
    output[top: top + imh, left: left + imw] = image
    return output


def build_transforms(args: Any, augment: bool):
    """
    Classification transform for action models that need full body shape.

    Letterbox is applied in the dataset before this transform. This transform
    only resizes, applies non-cropping augmentations and normalizes to 0-1.
    """
    import torchvision.transforms as T

    transforms = [
        T.Resize((args.imgsz, args.imgsz), interpolation=T.InterpolationMode.BILINEAR),
    ]
    if augment:
        if args.fliplr > 0:
            transforms.append(T.RandomHorizontalFlip(p=args.fliplr))
        if args.auto_augment:
            auto_augment = str(args.auto_augment).lower()
            interpolation = T.InterpolationMode.BILINEAR
            if auto_augment == "randaugment":
                transforms.append(T.RandAugment(interpolation=interpolation))
            elif auto_augment == "augmix":
                transforms.append(T.AugMix(interpolation=interpolation))
            elif auto_augment == "autoaugment":
                transforms.append(T.AutoAugment(interpolation=interpolation))
            else:
                raise ValueError(f"不支持的 auto_augment: {args.auto_augment}")
        elif args.hsv_v > 0 or args.hsv_s > 0 or args.hsv_h > 0:
            transforms.append(
                T.ColorJitter(
                    brightness=args.hsv_v,
                    contrast=args.hsv_v,
                    saturation=args.hsv_s,
                    hue=args.hsv_h,
                )
            )
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=torch.tensor((0.0, 0.0, 0.0)), std=torch.tensor((1.0, 1.0, 1.0))),
    ])
    if augment and args.erasing > 0:
        transforms.append(T.RandomErasing(p=args.erasing, inplace=True))
    return T.Compose(transforms)


class LetterBoxClassificationDataset(ClassificationDataset):
    def __init__(self, root: str, args: Any, augment: bool = False, prefix: str = "") -> None:
        super().__init__(root=root, args=args, augment=augment, prefix=prefix)
        self.augment = augment
        self.torch_transforms = build_transforms(args=args, augment=augment)

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

        im = letter_box(im)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}


class LetterBoxCropStyleClassificationDataset(LetterBoxClassificationDataset):
    def __getitem__(self, i: int) -> dict:
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:
            im = cv2.imread(f)
        if im is None:
            raise FileNotFoundError(f"无法读取图片: {f}")

        if self.augment:
            im = random_crop_style(im)
        im = letter_box(im)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}


class LetterBoxClassificationValidator(ClassificationValidator):
    def build_dataset(self, img_path: str) -> LetterBoxClassificationDataset:
        return LetterBoxClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)


class LetterBoxClassificationTrainer(ClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return LetterBoxClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_validator(self):
        self.loss_names = ["loss"]
        return LetterBoxClassificationValidator(
            self.test_loader,
            self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


class LetterBoxCropStyleClassificationTrainer(LetterBoxClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        if mode == "train":
            return LetterBoxCropStyleClassificationDataset(root=img_path, args=self.args, augment=True, prefix=mode)
        return LetterBoxClassificationDataset(root=img_path, args=self.args, augment=False, prefix=mode)
