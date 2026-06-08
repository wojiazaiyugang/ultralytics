from copy import copy
from typing import Any

import comet_ml
import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="student-action-classify")


def letter_box(image: np.ndarray) -> np.ndarray:
    """
    长边不变，短边居中补齐到与长边一致（方图）
    :return: 补边后的Frame
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
    学生动作分类依赖完整人体轮廓，训练和验证都不做 center crop / random crop。
    letterbox 在 __getitem__ 中完成，这里只负责缩放、轻量增强和归一化。
    """
    import torchvision.transforms as T

    transforms = [
        T.Resize((args.imgsz, args.imgsz), interpolation=T.InterpolationMode.BILINEAR),
    ]
    if augment:
        if args.fliplr > 0:
            transforms.append(T.RandomHorizontalFlip(p=args.fliplr))
        if args.hsv_v > 0 or args.hsv_s > 0 or args.hsv_h > 0:
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
        self.torch_transforms = build_transforms(args=args, augment=augment)

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
        if im is None:
            raise FileNotFoundError(f"无法读取图片: {f}")
        im = letter_box(im)
        # Convert NumPy array to PIL image
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


def main():
    model = YOLO("yolo11s-cls.pt")

    train_kwargs = dict(
        data="/DATA/yujiannan/Datasets/process_20260422_updating_limited",
        batch=96,
        epochs=300,
        imgsz=224,
        exist_ok=False,
        project="logs/student_action_classify",
        name="38",
        erasing=0.0,
        auto_augment=None,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.15,
        hsv_v=0.15,
    )

    PREPROCESS = "letterbox"  # 可选: "center_crop", "letterbox"

    if PREPROCESS == "center_crop":
        # Ultralytics 默认分类预处理：train 使用轻量 RandomResizedCrop，val/predict 使用 Resize + CenterCrop。
        train_kwargs.update(scale=0.1)
    elif PREPROCESS == "letterbox":
        # 不使用 RandomResizedCrop / RandomErasing，避免裁掉腿、头、桌面边界后破坏站立判断。
        train_kwargs.update(trainer=LetterBoxClassificationTrainer, scale=0.0)
    else:
        raise ValueError(f"不支持的 PREPROCESS: {PREPROCESS}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
