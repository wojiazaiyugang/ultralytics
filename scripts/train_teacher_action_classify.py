import comet_ml

from ultralytics import YOLO

from crop_style_classification import CropStyleClassificationTrainer
from letterbox_classification import LetterBoxClassificationTrainer


comet_ml.login(api_key="gq76e4j6CHnkcgarANUr5uXjV",
               workspace="wojiazaiyugang",
               project_name="teacher-action-classify")

PREPROCESS = "letterbox"  # 可选: "center_crop", "crop_style", "letterbox"


def main():
    model = YOLO("yolo11s-cls.pt")

    train_kwargs = dict(
        data="/DATA/yujiannan/Datasets/process_20260414_updating_limited",
        batch=16,
        epochs=300,
        imgsz=224,
        patience=100,
        exist_ok=False,
        project="logs/teacher_action_classify",
        name="29",
        dropout=0.1,
        weight_decay=0.001,
        cos_lr=False,
        erasing=0,
        auto_augment="randaugment",
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    if PREPROCESS == "center_crop":
        # Ultralytics 默认分类预处理：train 使用轻量 RandomResizedCrop，val/predict 使用 Resize + CenterCrop。
        train_kwargs.update(scale=0.5)
    elif PREPROCESS == "crop_style":
        # 只在训练集上随机改变 crop 风格，验证和测试仍保持默认 center crop。
        train_kwargs.update(trainer=CropStyleClassificationTrainer, scale=0.5)
    elif PREPROCESS == "letterbox":
        # 不使用 RandomResizedCrop，避免裁掉老师全身轮廓后破坏坐/站/板书判断。
        train_kwargs.update(trainer=LetterBoxClassificationTrainer, scale=0.0)
    else:
        raise ValueError(f"不支持的 PREPROCESS: {PREPROCESS}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
