# CNN Robustness & Data Pipeline Integrity Testing

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![EfficientNet](https://img.shields.io/badge/model-EfficientNetB0-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-preprocessing-green)

A model validation project that treats a transfer-learning image classifier as something to be **stress-tested**, not just trained: quantifying how much performance degrades under real-world input quality issues, and auditing whether the data pipeline itself introduces bias.

## Why robustness testing, not just accuracy

A model that scores well on a clean validation set can still fail in production if the inputs it receives are noisier, differently lit, or lower resolution than the training data. This project treats that gap as the primary object of study: the headline result is not "how accurate is the model" but "how much does accuracy fall off a cliff, and under what conditions".

## What was tested

**Transfer learning base model:** EfficientNetB0, fine-tuned on the target image classes.

**Degradation stress test:** the fine-tuned model was evaluated on the same test set under two input conditions:

| Input condition | Accuracy |
|---|---|
| RGB (normal colour input) | 91.5% |
| Grayscale (colour information removed) | 79.3% |

The **12.2-point drop** when colour information is removed is treated as a quantified risk finding, not a footnote: it tells you the model is relying on colour cues more than a robust model should, and flags a real deployment risk if the system will ever see grayscale, low-light, or colour-degraded camera input in production.

**Data pipeline audit:** multiple augmentation strategies (rotation, flipping, colour jitter, zoom) were evaluated for whether they introduced distributional bias into the training set relative to the original class balance, checking that augmentation improves generalisation without silently over- or under-representing any class.

**Training stability monitoring:** `ReduceLROnPlateau` was used to track and document training instability (loss plateaus, oscillation), a practical parallel to the post-deployment performance surveillance and drift detection required under AI governance frameworks such as the NIST AI RMF's Measure function.

## Repository structure

```
cnn-robustness-testing/
  README.md
  requirements.txt
  src/
    train.py              # transfer-learning training script (EfficientNetB0)
    robustness_eval.py     # RGB vs grayscale (and other) degradation testing
    pipeline_audit.py      # augmentation strategy bias audit
```

## Getting started

```bash
git clone https://github.com/Nancy-MK/cnn-robustness-testing.git
cd cnn-robustness-testing
pip install -r requirements.txt

# 1. Train the base model
python src/train.py --data-dir data/ --epochs 15

# 2. Run the robustness/degradation stress test
python src/robustness_eval.py --model artifacts/model.keras --data-dir data/test

# 3. Audit the augmentation pipeline for distributional bias
python src/pipeline_audit.py --data-dir data/train
```

Images are expected to be organised as `data/<split>/<class_name>/*.jpg`, the standard `image_dataset_from_directory` layout.

## Skills demonstrated

- Transfer learning and fine-tuning (EfficientNetB0)
- Robustness / stress testing under input degradation
- Data pipeline and augmentation bias auditing
- Training stability monitoring and drift-relevant documentation

## Tech stack

Python, TensorFlow/Keras, OpenCV, NumPy

## Licence

Developed for academic purposes. All rights reserved (c) Nancy Kamal.
