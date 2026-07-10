import comet_ml

from ultralytics import YOLO


comet_ml.login(
    api_key="gq76e4j6CHnkcgarANUr5uXjV",
    workspace="wojiazaiyugang",
    project_name="human-detection",
)


def main() -> None:
    # 实验28：冻结18的完整backbone，只让neck和检测头小步适配当前修标数据。
    # 目标是吸收角色修正，同时减少全模型漂移，优先保护小目标和站立学生召回。
    model = YOLO("/home/yujiannan/Projects/ultralytics/scripts/logs/human_detection/18/weights/best.pt")

    train_kwargs = dict(
        data="/DATA/yujiannan/Datasets/process_20251013_20260511_updating/data.yaml",
        batch=4,
        epochs=120,
        imgsz=1280,
        patience=50,
        exist_ok=False,
        project="/home/yujiannan/Projects/ultralytics/scripts/logs/human_detection",
        name="28",
        save_period=2,
        optimizer="AdamW",
        lr0=2e-5,
        lrf=0.1,
        warmup_epochs=1.0,
        warmup_bias_lr=0.0,
        weight_decay=0.0005,
        cos_lr=True,
        freeze=11,
        box=7.5,
        cls=0.75,
        dfl=1.5,
        mosaic=0.35,
        close_mosaic=30,
        mixup=0.0,
        copy_paste=0.0,
        scale=0.35,
        translate=0.05,
        degrees=2.0,
        shear=2.0,
        perspective=0.0001,
        erasing=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        seed=0,
    )

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
