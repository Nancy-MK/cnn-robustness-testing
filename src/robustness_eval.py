"""
robustness_eval.py

Stress-tests a trained model against degraded input conditions (currently:
RGB vs. grayscale) and reports the accuracy drop as a quantified risk
finding, the same logic used in model validation / robustness engagements.

Usage:
    python src/robustness_eval.py --model artifacts/model.keras --data-dir data/test
"""
import argparse

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)


def to_grayscale_but_3_channel(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale, then replicate it across 3 channels
    so the model's input shape is unchanged, only the colour information
    is removed."""
    gray = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_RGB2GRAY)
    return np.stack([gray, gray, gray], axis=-1).astype("float32")


def evaluate(model, dataset, name: str):
    loss, acc = model.evaluate(dataset, verbose=0)
    print(f"{name}: accuracy={acc:.4f}, loss={loss:.4f}")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness under input degradation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    rgb_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir, image_size=IMG_SIZE, batch_size=args.batch_size
    )

    gray_ds = rgb_ds.map(
        lambda x, y: (
            tf.numpy_function(
                lambda batch: np.stack([to_grayscale_but_3_channel(img) for img in batch]),
                [x],
                tf.float32,
            ),
            y,
        )
    )

    rgb_acc = evaluate(model, rgb_ds, "RGB (normal input)")
    gray_acc = evaluate(model, gray_ds, "Grayscale (degraded input)")

    drop = rgb_acc - gray_acc
    print(f"\nAccuracy drop under grayscale degradation: {drop:.4f} ({drop * 100:.1f} points)")
    if drop > 0.05:
        print(
            "RISK FINDING: accuracy drop exceeds 5 points. The model may be over-reliant "
            "on colour cues; consider colour-invariant augmentation during training if the "
            "deployment environment cannot guarantee consistent, well-lit colour input."
        )


if __name__ == "__main__":
    main()
