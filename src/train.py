"""
train.py

Fine-tunes an EfficientNetB0 backbone (ImageNet weights) on a target image
classification dataset via transfer learning, with ReduceLROnPlateau used to
monitor and respond to training instability.

Usage:
    python src/train.py --data-dir data/ --epochs 15
"""
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

IMG_SIZE = (224, 224)


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3), pooling="avg"
    )
    base_model.trainable = False  # start with a frozen backbone

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 via transfer learning.")
    parser.add_argument("--data-dir", required=True, help="Root dir with train/ and val/ subfolders.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", image_size=IMG_SIZE, batch_size=args.batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val", image_size=IMG_SIZE, batch_size=args.batch_size
    )
    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")

    # Light augmentation applied only at train time.
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))

    model = build_model(num_classes=len(class_names))

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")

    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\nFinal validation accuracy: {final_val_acc:.4f}")
    print(f"Model saved to {out_dir}/model.keras")


if __name__ == "__main__":
    main()
