import tensorflow as tf
from keras import layers
from keras.utils import image_dataset_from_directory
from PIL import Image
from pathlib import Path

# 学習前に画像形式をJPEGへ統一する
def convert_to_jpg(folder: Path, delete_original=False):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() not in [".jpg", ".jpeg"]:
            try:
                img = Image.open(path)
                new_path = path.with_suffix(".jpg")

                img.convert("RGB").save(new_path, "JPEG")
                print(f"変換: {path.name} → {new_path.name}")

                if delete_original:
                    path.unlink()

            except Exception as e:
                print(f"失敗: {path} ({e})")

# 訓練データと検証データを作成する
def create_datasets(
    data_dir: Path,
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 123,
    val_split: float = 0.2
):
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        image_size=img_size,
        batch_size=batch_size,
        subset='training',
        seed=seed
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        image_size=img_size,
        batch_size=batch_size,
        subset='validation',
        seed=seed
    )

    return train_ds, val_ds

# データ拡張によって過学習を抑制する
def get_data_augmentation() -> tf.keras.Sequential:
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"), #左右反転
        layers.RandomRotation(0.1),     # 撮影角度のばらつきを再現
        layers.RandomZoom(0.2),         #画像拡大
    ])
    return data_augmentation

# 訓練データ数と検証データ数を確認
def count_dataset_size(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset
) -> None:
    train_count = sum(1 for _ in train_ds.unbatch()) 
    val_count =sum(1 for _ in val_ds.unbatch())
    print("訓練データ数:", train_count)
    print("検証データ数:", val_count)
