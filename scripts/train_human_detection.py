import comet_ml

from ultralytics import YOLO


comet_ml.login(
    api_key="gq76e4j6CHnkcgarANUr5uXjV",
    workspace="wojiazaiyugang",
    project_name="human-detection",
)


def main() -> None:
    # 实验27：过夜长实验，从当前线上候选18继续微调当前修标数据。
    # 目标是吸收修标后的更高质量标注，同时保持18已有的 student recall 和人数统计稳定性。
    model = YOLO("/home/yujiannan/Projects/ultralytics/scripts/logs/human_detection/18/weights/best.pt")

    train_kwargs = dict(
        data="/DATA/yujiannan/Datasets/process_20251013_20260511_updating/data.yaml",
        batch=4,
        epochs=300,
        imgsz=1280,
        patience=120,
        exist_ok=False,
        project="/home/yujiannan/Projects/ultralytics/scripts/logs/human_detection",
        name="27",
        save_period=10,
        optimizer="AdamW",
        lr0=2e-5,
        lrf=0.05,
        warmup_epochs=1.0,
        warmup_bias_lr=0.0,
        weight_decay=0.0005,
        cos_lr=True,
        box=7.5,
        cls=1.0,
        dfl=1.5,
        mosaic=0.35,
        close_mosaic=60,
        mixup=0.0,
        copy_paste=0.0,
        scale=0.35,
        translate=0.05,
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
