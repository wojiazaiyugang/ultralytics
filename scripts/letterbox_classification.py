import os
from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import WeightedRandomSampler

from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
from ultralytics import YOLO
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first

from crop_style_classification import random_crop_style


STUDENT_ACTION_CLASSES = [
    "blur",
    "computer",
    "discuss",
    "head_down_other",
    "head_down_read_write",
    "head_up",
    "like_phone",
    "pad",
    "phone",
    "sleep",
    "stand",
]
FOCUS_CLASSES = {"stand", "head_up", "computer", "pad", "head_down_read_write", "discuss"}
HEADUP_CLASSES = {"stand", "head_up"}
BUSINESS_CLASS_MULTIPLIERS = {
    "blur": 1.10,
    "computer": 1.10,
    "head_down_other": 0.90,
    "head_down_read_write": 1.25,
    "head_up": 1.05,
    "like_phone": 1.15,
    "pad": 1.15,
    "phone": 1.25,
    "stand": 1.05,
}


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


def normalize_names(names: dict[Any, Any] | list[Any]) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def class_counts_from_dataset(data_dir: str | Path, names: dict[int, str]) -> dict[str, int]:
    train_dir = Path(data_dir) / "train"
    counts = {}
    for class_name in names.values():
        class_dir = train_dir / class_name
        counts[class_name] = sum(
            1
            for path in class_dir.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ) if class_dir.exists() else 0
    return counts


def business_class_weights(data_dir: str | Path, names: dict[int, str]) -> list[float]:
    counts = class_counts_from_dataset(data_dir=data_dir, names=names)
    valid_counts = [count for count in counts.values() if count > 0]
    if not valid_counts:
        return [1.0 for _ in names]

    total = sum(valid_counts)
    class_count = len(valid_counts)
    weights = []
    for index in sorted(names):
        class_name = names[index]
        count = max(counts.get(class_name, 0), 1)
        # Inverse-frequency weights are square-rooted and clipped to avoid turning noisy rare classes dominant.
        weight = (total / (class_count * count)) ** 0.5
        weight = min(max(weight, 0.65), 1.80)
        weight *= BUSINESS_CLASS_MULTIPLIERS.get(class_name, 1.0)
        weights.append(float(weight))
    mean_weight = sum(weights) / len(weights)
    return [min(max(weight / mean_weight, 0.65), 1.45) for weight in weights]


def imbalance_class_weights(
    data_dir: str | Path,
    names: dict[int, str],
    mode: str,
    power: float = 0.5,
    effective_beta: float = 0.9995,
    min_weight: float = 0.35,
    max_weight: float = 2.50,
    use_business_multipliers: bool = False,
) -> list[float]:
    counts = class_counts_from_dataset(data_dir=data_dir, names=names)
    valid_counts = [count for count in counts.values() if count > 0]
    if not valid_counts or mode == "none":
        return [1.0 for _ in names]

    total = sum(valid_counts)
    class_count = len(valid_counts)
    weights = []
    business_weights = business_class_weights(data_dir=data_dir, names=names) if mode == "business" else None
    for index in sorted(names):
        class_name = names[index]
        count = max(counts.get(class_name, 0), 1)
        if mode == "business":
            weight = business_weights[index]
        elif mode == "inverse":
            weight = (total / (class_count * count)) ** power
        elif mode == "effective":
            beta = min(max(effective_beta, 0.0), 0.999999)
            effective_num = 1.0 - (beta ** count)
            weight = (1.0 - beta) / max(effective_num, 1e-12)
        else:
            raise ValueError(f"不支持的类别权重模式: {mode}")

        if use_business_multipliers and mode != "business":
            weight *= BUSINESS_CLASS_MULTIPLIERS.get(class_name, 1.0)
        weights.append(float(weight))

    mean_weight = sum(weights) / len(weights)
    return [min(max(weight / mean_weight, min_weight), max_weight) for weight in weights]


def class_log_adjustments(data_dir: str | Path, names: dict[int, str]) -> list[float]:
    counts = class_counts_from_dataset(data_dir=data_dir, names=names)
    log_counts = []
    for index in sorted(names):
        count = max(counts.get(names[index], 0), 1)
        log_counts.append(float(np.log(count)))
    mean_log_count = sum(log_counts) / len(log_counts)
    return [value - mean_log_count for value in log_counts]


def sample_weights_from_dataset(
    dataset: ClassificationDataset,
    power: float = 0.5,
    min_weight: float = 0.35,
    max_weight: float = 2.50,
) -> tuple[list[float], np.ndarray, np.ndarray]:
    labels = np.asarray([int(sample[1]) for sample in dataset.samples], dtype=np.int64)
    if labels.size == 0:
        raise ValueError("训练数据为空，无法构建类别均衡采样器")

    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
    valid = counts > 0
    total = float(counts[valid].sum())
    class_count = float(valid.sum())
    class_weights = np.ones_like(counts, dtype=np.float64)
    class_weights[valid] = (total / (class_count * counts[valid])) ** power
    mean_weight = float(class_weights[valid].mean())
    if mean_weight > 0:
        class_weights[valid] = class_weights[valid] / mean_weight
    class_weights[valid] = np.clip(class_weights[valid], min_weight, max_weight)
    return class_weights[labels].astype(np.float64).tolist(), counts.astype(np.int64), class_weights


class StudentActionBusinessLoss:
    def __init__(
        self,
        names: dict[int, str],
        data_dir: str | Path,
        focus_gain: float = 0.35,
        headup_gain: float = 0.25,
        stand_gain: float = 0.10,
    ) -> None:
        class_names = [names[index] for index in sorted(names)]
        if class_names != STUDENT_ACTION_CLASSES:
            raise ValueError(f"学生动作类别顺序不符合预期: {class_names}")
        self.names = names
        self.focus_gain = focus_gain
        self.headup_gain = headup_gain
        self.stand_gain = stand_gain
        self.class_weights = business_class_weights(data_dir=data_dir, names=names)
        self.focus_indices = [index for index, name in names.items() if name in FOCUS_CLASSES]
        self.headup_indices = [index for index, name in names.items() if name in HEADUP_CLASSES]
        self.stand_indices = [index for index, name in names.items() if name == "stand"]

    @staticmethod
    def _binary_group_loss(probs: torch.Tensor, target: torch.Tensor, indices: list[int]) -> torch.Tensor:
        group_prob = probs[:, indices].sum(dim=1).clamp(min=1e-6, max=1 - 1e-6)
        group_target = torch.zeros_like(group_prob)
        for index in indices:
            group_target = torch.where(target == index, torch.ones_like(group_target), group_target)
        return -(group_target * group_prob.log() + (1 - group_target) * (1 - group_prob).log()).mean()

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        target = batch["cls"].long()
        class_weights = preds.new_tensor(self.class_weights)
        ce_loss = F.cross_entropy(preds, target, weight=class_weights, reduction="mean")
        probs = preds.softmax(dim=1)
        focus_loss = self._binary_group_loss(probs=probs, target=target, indices=self.focus_indices)
        headup_loss = self._binary_group_loss(probs=probs, target=target, indices=self.headup_indices)
        stand_loss = self._binary_group_loss(probs=probs, target=target, indices=self.stand_indices)
        loss = (
            ce_loss
            + self.focus_gain * focus_loss
            + self.headup_gain * headup_loss
            + self.stand_gain * stand_loss
        )
        return loss, loss.detach()


class StudentActionImbalanceBusinessLoss(StudentActionBusinessLoss):
    def __init__(
        self,
        names: dict[int, str],
        data_dir: str | Path,
        focus_gain: float = 0.20,
        headup_gain: float = 0.15,
        stand_gain: float = 0.08,
        class_weight_mode: str = "effective",
        class_weight_power: float = 0.5,
        class_weight_min: float = 0.35,
        class_weight_max: float = 2.50,
        effective_beta: float = 0.9995,
        logit_adjustment: float = 0.0,
        focal_gamma: float = 0.0,
        use_business_multipliers: bool = False,
    ) -> None:
        super().__init__(
            names=names,
            data_dir=data_dir,
            focus_gain=focus_gain,
            headup_gain=headup_gain,
            stand_gain=stand_gain,
        )
        self.class_weights = imbalance_class_weights(
            data_dir=data_dir,
            names=names,
            mode=class_weight_mode,
            power=class_weight_power,
            effective_beta=effective_beta,
            min_weight=class_weight_min,
            max_weight=class_weight_max,
            use_business_multipliers=use_business_multipliers,
        )
        self.logit_adjustment = logit_adjustment
        self.log_adjustments = class_log_adjustments(data_dir=data_dir, names=names)
        self.focal_gamma = focal_gamma

    def _classification_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits
        if self.logit_adjustment:
            adjustments = logits.new_tensor(self.log_adjustments)
            adjusted_logits = logits + (self.logit_adjustment * adjustments)

        class_weights = logits.new_tensor(self.class_weights)
        ce = F.cross_entropy(adjusted_logits, target, weight=class_weights, reduction="none")
        if self.focal_gamma > 0:
            probs = adjusted_logits.softmax(dim=1)
            pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
            ce = ce * ((1.0 - pt) ** self.focal_gamma)

        denom = class_weights.gather(0, target).sum().clamp(min=1e-6)
        return ce.sum() / denom

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        target = batch["cls"].long()
        ce_loss = self._classification_loss(logits=preds, target=target)
        probs = preds.softmax(dim=1)
        focus_loss = self._binary_group_loss(probs=probs, target=target, indices=self.focus_indices)
        headup_loss = self._binary_group_loss(probs=probs, target=target, indices=self.headup_indices)
        stand_loss = self._binary_group_loss(probs=probs, target=target, indices=self.stand_indices)
        loss = (
            ce_loss
            + self.focus_gain * focus_loss
            + self.headup_gain * headup_loss
            + self.stand_gain * stand_loss
        )
        return loss, loss.detach()


def normalize_probabilities(outputs: torch.Tensor) -> torch.Tensor:
    outputs = outputs.float()
    row_sums = outputs.sum(dim=1)
    if bool(torch.all(outputs >= 0)) and torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.1):
        return outputs
    return outputs.softmax(dim=1)


class StudentActionDistillBusinessLoss(StudentActionBusinessLoss):
    def __init__(
        self,
        names: dict[int, str],
        data_dir: str | Path,
        teacher_weights: list[str | Path],
        teacher_ratios: list[float] | None = None,
        distill_alpha: float = 0.35,
        distill_temperature: float = 2.0,
        focus_gain: float = 0.35,
        headup_gain: float = 0.25,
        stand_gain: float = 0.10,
    ) -> None:
        super().__init__(
            names=names,
            data_dir=data_dir,
            focus_gain=focus_gain,
            headup_gain=headup_gain,
            stand_gain=stand_gain,
        )
        if not teacher_weights:
            raise ValueError("启用蒸馏训练时必须提供 teacher_weights")
        if teacher_ratios is None:
            teacher_ratios = [1.0 / len(teacher_weights) for _ in teacher_weights]
        if len(teacher_ratios) != len(teacher_weights):
            raise ValueError(f"teacher_ratios 数量必须和 teacher_weights 一致: {len(teacher_ratios)} != {len(teacher_weights)}")
        ratio_sum = sum(teacher_ratios)
        if ratio_sum <= 0:
            raise ValueError("teacher_ratios 之和必须大于 0")

        self.teacher_weights = [Path(weight) for weight in teacher_weights]
        self.teacher_ratios = [float(ratio / ratio_sum) for ratio in teacher_ratios]
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature
        self.teacher_nets = []
        self.teacher_device = None
        expected_classes = [names[index] for index in sorted(names)]
        for weight in self.teacher_weights:
            yolo = YOLO(weight.as_posix())
            teacher_names = normalize_names(yolo.names)
            teacher_classes = [teacher_names[index] for index in sorted(teacher_names)]
            if teacher_classes != expected_classes:
                raise ValueError(f"teacher 类别顺序不符合学生动作分类: {weight} {teacher_classes}")
            net = yolo.model.eval()
            for parameter in net.parameters():
                parameter.requires_grad_(False)
            self.teacher_nets.append(net)

    @staticmethod
    def _model_probabilities(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        return normalize_probabilities(outputs)

    def _move_teachers(self, device: torch.device) -> None:
        if self.teacher_device == device:
            return
        self.teacher_nets = [net.to(device).eval() for net in self.teacher_nets]
        self.teacher_device = device

    def _teacher_probs(self, images: torch.Tensor) -> torch.Tensor:
        self._move_teachers(images.device)
        teacher_images = images.float()
        probs = None
        with torch.inference_mode():
            for net, ratio in zip(self.teacher_nets, self.teacher_ratios):
                current = self._model_probabilities(net(teacher_images)).detach()
                probs = current * ratio if probs is None else probs + current * ratio
        if probs is None:
            raise RuntimeError("没有可用 teacher 概率")
        return probs.clamp(min=1e-6, max=1.0)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        student_logits = preds[1] if isinstance(preds, (list, tuple)) else preds
        base_loss, _ = super().__call__(student_logits, batch)
        teacher_probs = self._teacher_probs(batch["img"])
        temperature = max(float(self.distill_temperature), 1e-6)
        teacher_soft = teacher_probs.pow(1.0 / temperature)
        teacher_soft = teacher_soft / teacher_soft.sum(dim=1, keepdim=True).clamp(min=1e-6)
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            teacher_soft,
            reduction="batchmean",
        ) * (temperature ** 2)
        loss = base_loss + self.distill_alpha * distill_loss
        return loss, loss.detach()


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


class WeightedSamplerLetterBoxClassificationTrainer(LetterBoxClassificationTrainer):
    sampler_power = 0.5
    sampler_min_weight = 0.35
    sampler_max_weight = 2.50

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train" or rank != -1:
            return super().get_dataloader(dataset_path=dataset_path, batch_size=batch_size, rank=rank, mode=mode)

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        sample_weights, counts, class_weights = sample_weights_from_dataset(
            dataset=dataset,
            power=self.sampler_power,
            min_weight=self.sampler_min_weight,
            max_weight=self.sampler_max_weight,
        )
        names = normalize_names(self.data["names"])
        weight_msg = ", ".join(
            f"{names.get(index, index)}:{int(count)}->{class_weights[index]:.2f}"
            for index, count in enumerate(counts)
            if count > 0
        )
        LOGGER.info(f"Weighted sampler enabled, power={self.sampler_power}, class weights: {weight_msg}")

        nd = torch.cuda.device_count()
        workers = min(os.cpu_count() // max(nd, 1), self.args.workers)
        batch_size = min(batch_size, len(dataset))
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            sampler=sampler,
            prefetch_factor=4 if workers > 0 else None,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=self.args.compile and len(dataset) % batch_size != 0,
        )


class StudentActionBusinessLossTrainer(LetterBoxClassificationTrainer):
    focus_gain = 0.35
    headup_gain = 0.25
    stand_gain = 0.10

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        names = normalize_names(self.data["names"])
        model.criterion = StudentActionBusinessLoss(
            names=names,
            data_dir=self.args.data,
            focus_gain=self.focus_gain,
            headup_gain=self.headup_gain,
            stand_gain=self.stand_gain,
        )
        return model


class StudentActionWeightedSamplerBusinessLossTrainer(WeightedSamplerLetterBoxClassificationTrainer):
    focus_gain = 0.35
    headup_gain = 0.25
    stand_gain = 0.10

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        names = normalize_names(self.data["names"])
        model.criterion = StudentActionBusinessLoss(
            names=names,
            data_dir=self.args.data,
            focus_gain=self.focus_gain,
            headup_gain=self.headup_gain,
            stand_gain=self.stand_gain,
        )
        return model


class StudentActionDistillBusinessLossTrainer(StudentActionBusinessLossTrainer):
    teacher_weights: list[str] = []
    teacher_ratios: list[float] | None = None
    distill_alpha = 0.35
    distill_temperature = 2.0

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = ClassificationTrainer.get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        names = normalize_names(self.data["names"])
        model.criterion = StudentActionDistillBusinessLoss(
            names=names,
            data_dir=self.args.data,
            teacher_weights=self.teacher_weights,
            teacher_ratios=self.teacher_ratios,
            distill_alpha=self.distill_alpha,
            distill_temperature=self.distill_temperature,
            focus_gain=self.focus_gain,
            headup_gain=self.headup_gain,
            stand_gain=self.stand_gain,
        )
        return model


class StudentActionImbalanceBusinessLossTrainer(StudentActionBusinessLossTrainer):
    class_weight_mode = "effective"
    class_weight_power = 0.5
    class_weight_min = 0.35
    class_weight_max = 2.50
    effective_beta = 0.9995
    logit_adjustment = 0.0
    focal_gamma = 0.0
    use_business_multipliers = False

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = ClassificationTrainer.get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        names = normalize_names(self.data["names"])
        model.criterion = StudentActionImbalanceBusinessLoss(
            names=names,
            data_dir=self.args.data,
            focus_gain=self.focus_gain,
            headup_gain=self.headup_gain,
            stand_gain=self.stand_gain,
            class_weight_mode=self.class_weight_mode,
            class_weight_power=self.class_weight_power,
            class_weight_min=self.class_weight_min,
            class_weight_max=self.class_weight_max,
            effective_beta=self.effective_beta,
            logit_adjustment=self.logit_adjustment,
            focal_gamma=self.focal_gamma,
            use_business_multipliers=self.use_business_multipliers,
        )
        return model


class LetterBoxCropStyleClassificationTrainer(LetterBoxClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        if mode == "train":
            return LetterBoxCropStyleClassificationDataset(root=img_path, args=self.args, augment=True, prefix=mode)
        return LetterBoxClassificationDataset(root=img_path, args=self.args, augment=False, prefix=mode)
