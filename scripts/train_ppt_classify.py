import comet_ml

from ultralytics import YOLO

from letterbox_classification import LetterBoxClassificationTrainer


comet_ml.login(
    api_key="gq76e4j6CHnkcgarANUr5uXjV",
    workspace="wojiazaiyugang",
    project_name="ppt-classify",
)

PREPROCESS = "letterbox"  # 可选: "letterbox", "center_crop"


def main():
    model = YOLO("yolo11n-cls.pt")

    train_kwargs = dict(
        data="/DATA/yujiannan/Datasets/20260623_updating",
        batch=64,
        epochs=300,
        imgsz=224,
        patience=50,
        exist_ok=False,
        project="logs/ppt_classify",
        name="2",
        dropout=0.1,
        weight_decay=0.01,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        warmup_epochs=3.0,
        warmup_bias_lr=0.0,
        erasing=0.0,
        auto_augment=None,
        fliplr=0.0,
        hsv_h=0.0,
        hsv_s=0.1,
        hsv_v=0.1,
    )

    if PREPROCESS == "letterbox":
        # 整屏分类依赖浏览器栏、播放器控件、PPT 边界等全局结构，避免中心裁剪裁掉关键信息。
        train_kwargs.update(trainer=LetterBoxClassificationTrainer, scale=0.0)
    elif PREPROCESS == "center_crop":
        train_kwargs.update(scale=0.1)
    else:
        raise ValueError(f"不支持的 PREPROCESS: {PREPROCESS}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
