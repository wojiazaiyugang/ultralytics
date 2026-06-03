from copy import copy
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


def letter_box(image: np.ndarray, color: int = 114) -> np.ndarray:
    """
    将原图按长边补成方图，不裁剪人体内容。
    后续 transform 只负责 resize/增强，不能再 center crop。
    """
    if image is None:
        raise ValueError("cv2.imread 读取图片失败")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    imh, imw = image.shape[:2]
    side = max(imh, imw)
    top = (side - imh) // 2
    left = (side - imw) // 2

    output = np.full((side, side, 3), color, dtype=image.dtype)
    output[top: top + imh, left: left + imw] = image
    return output


def build_letterbox_transforms(args: Any, augment: bool):
    """
    letterbox 已经保留完整人体，这里只 resize，不再随机裁剪/中心裁剪。
    继续保留水平翻转、颜色扰动和随机擦除等不会破坏人体整体形态的增强。
    """
    import torchvision.transforms as T

    imgsz = args.imgsz[0] if isinstance(args.imgsz, (list, tuple)) else args.imgsz
    transforms = [T.Resize((imgsz, imgsz), interpolation=T.InterpolationMode.BILINEAR)]
    if augment:
        if args.fliplr > 0.0:
            transforms.append(T.RandomHorizontalFlip(p=args.fliplr))
        if args.flipud > 0.0:
            transforms.append(T.RandomVerticalFlip(p=args.flipud))
        if args.auto_augment is None:
            transforms.append(T.ColorJitter(brightness=args.hsv_v,
                                            contrast=args.hsv_v,
                                            saturation=args.hsv_s,
                                            hue=args.hsv_h))
    transforms.extend([T.ToTensor(), T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))])
    if augment and args.erasing > 0.0:
        transforms.append(T.RandomErasing(p=args.erasing, inplace=True))
    return T.Compose(transforms)


class LetterBoxClassificationDataset(ClassificationDataset):
    def __init__(self, *args: Any, augment: bool = False, **kwargs: Any) -> None:
        self.args = kwargs.get("args")
        if self.args is None:
            raise ValueError("LetterBoxClassificationDataset 必须通过 args=... 传入训练参数")
        super().__init__(*args, augment=augment, **kwargs)
        self.torch_transforms = build_letterbox_transforms(self.args, augment=augment)

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
        im = letter_box(im)
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}


class LetterBoxClassificationValidator(ClassificationValidator):
    def build_dataset(self, img_path: str) -> ClassificationDataset:
        return LetterBoxClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size: int):
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)


class Trainer(ClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return LetterBoxClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_validator(self):
        self.loss_names = ["loss"]
        return LetterBoxClassificationValidator(self.test_loader,
                                                self.save_dir,
                                                args=copy(self.args),
                                                _callbacks=self.callbacks)


def train():
    try:
        import comet_ml
        comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
                       workspace="wojiazaiyugang",
                       project_name="student-action-classify")
    except ImportError:
        print("comet_ml 未安装，跳过 Comet 记录")

    model = YOLO("yolo11s-cls.pt")
    return model.train(
        data="/DATA/yujiannan/Datasets/process_20260422_updating_limited",
        batch=96,
        epochs=300,
        imgsz=224,
        exist_ok=False,
        project="logs/student_action_classify",
        name="36",
        trainer=Trainer,
        scale=0.0,
        erasing=0.0,
        auto_augment=None,
        fliplr=0.5,
    )


if __name__ == "__main__":
    train()
