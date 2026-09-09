# Experimental Methodology & Evaluation Plan

## 1. Models Compared (Disease Classification)
1. **CNN Baseline** — trained from scratch (`app/ml/train_cnn.py`)
2. **Image-only Transfer Learning** — MobileNetV2 (`app/ml/train_transfer.py`)
3. **Proposed Multimodal Approach** — MobileNetV2 output fused with the
   field-context Random Forest output (`app/services/fusion.py`)

All three should be trained/evaluated on the **same** train/val/test
split (produced once by `app/ml/dataset_utils.py`) so comparisons are
fair.

## 2. Metrics

| Task | Metrics | Script |
|---|---|---|
| Disease classification | Accuracy, Precision, Recall, F1-score (weighted + per-class), Confusion Matrix | `app/ml/evaluate.py` |
| Severity classification | Accuracy, F1-score (if you manually label a small validation subset — see Section 4) | Extend `evaluate.py` or a notebook |
| Affected-area regression (optional, if you manually annotate a subset) | MAE, RMSE | Compare `estimate_severity()` output vs manual annotation |
| Field risk classification | Accuracy, F1-score (on the synthetic held-out test set — see caveat below) | Printed by `train_field_model.py` |

## 3. How to Honestly Test Whether Field Context Improves Results
This is the most important — and most commonly faked — experiment in
this kind of project. Do NOT simply assert improvement. Instead:

1. Build a small **manually curated test set** (e.g. 30-60 images you
   personally select, one per class where possible) where you also
   record realistic field-context values (you can look up typical
   weather for the region/season the photos were taken, or construct
   plausible scenarios and clearly label them as such).
2. Run the **image-only** pipeline (Model 2) and record predictions.
3. Run the **full multimodal + fusion** pipeline (Model 3) and record
   predictions, including `final_confidence` and whether the fusion
   step changed the top-1 disease call or only the confidence score
   (note: in the current fusion design in `fusion.py`, fusion adjusts
   *confidence*, not the top-1 label — be precise about this
   distinction in your report; don't claim label-accuracy improvement
   if only confidence calibration improved).
4. Report both: (a) whether top-1 accuracy changed, and (b) whether
   confidence calibration improved (e.g., were high-confidence wrong
   predictions reduced, or did the model appropriately lower confidence
   on cases where field conditions contradicted the visual diagnosis).
5. If results show **no accuracy improvement**, report that honestly.
   A well-reasoned negative or neutral result, with a clear explanation
   of why the current fusion strategy affects confidence calibration
   rather than raw label accuracy, is stronger for a viva than a
   fabricated improvement claim.

## 4. Evaluating Severity Without Ground Truth
Since PlantVillage/PlantDoc do not provide lesion masks or
affected-area ground truth:
- Option A (recommended, low-effort): 2-3 team members independently
  visually estimate "rough affected area %" on a small (~30-40 image)
  sample, average their estimates as a pseudo-ground-truth, and compute
  MAE/RMSE between `estimate_severity()` output and this average.
  Clearly label this as a small-scale, subjective pseudo-validation in
  your report — not a rigorous ground truth.
- Option B (if time permits): use free tools like LabelMe/CVAT to draw
  rough lesion polygons on a subset and compute IoU-based area estimates
  as a stronger pseudo-ground-truth.

## 5. Evaluating Field Risk Without a Labeled Risk Dataset
Report the Random Forest's performance on its own synthetic held-out
test split (already printed by `train_field_model.py`) as evidence the
model learned the intended domain rules, and explicitly state in your
report that this does NOT constitute validation against real-world
disease-outbreak data, since no such public dataset was available for
this MVP. Suggest real validation as future work (partnering with an
agricultural extension office or using historical outbreak + weather
records if accessible).

## 6. Expected Results (write these AFTER running real experiments)
Do not pre-fill numbers here before training — replace this section
with your actual measured accuracy/F1/confusion matrices once training
is complete. A template table:

| Model | Test Accuracy | Weighted F1 |
|---|---|---|
| CNN Baseline | _(fill in)_ | _(fill in)_ |
| MobileNetV2 Transfer | _(fill in)_ | _(fill in)_ |
| Field-Risk Random Forest | _(fill in)_ | _(fill in)_ |

## 7. Leaf-Segmentation Improvement Experiment (v2) — RESULT: REJECTED

Per the requirement to "evaluate whether improved leaf segmentation
provides better affected-area estimation and implement it only if
testing demonstrates reliable improvement," we built a synthetic
ground-truth benchmark (`app/ml/evaluate_severity_segmentation.py`) —
images with a KNOWN, exactly-painted affected-area percentage — and
compared the production method (v1: fixed HSV thresholds) against a
candidate improvement (v2: LAB a*-channel + automatic Otsu
thresholding, chosen because the a* channel directly encodes the
green-vs-red/brown axis and Otsu self-calibrates per image instead of
relying on fixed cutoffs).

**Actual measured result (n=60 synthetic images, seed=42):**

| Method | MAE | RMSE | Avg. false-positive area on healthy images |
|---|---|---|---|
| v1 — HSV fixed-threshold (production) | 11.53% | 14.50% | 0.00% |
| v2 — LAB a* + Otsu (candidate) | 10.00% | 31.62% | 100.00% |

v2's mean absolute error was only 13.3% relatively better than v1 —
below our pre-registered 20% adoption bar — and, critically, v2
catastrophically fails on healthy leaves: Otsu's method always finds
*some* binary split point even in a near-uniform green image (it has
no concept of "no lesion present"), so it flags roughly half of every
healthy leaf as diseased. v1's fixed thresholds correctly report ~0%
on healthy leaves.

**Decision: v1 (the original HSV heuristic) was kept in production.**
`app/services/severity.py` was NOT changed. This is documented here as
a real, honest negative result rather than silently discarded — it is
a valid experimental finding for the report: "a per-image adaptive
thresholding approach was tested and rejected because it lacks a
no-lesion baseline case, unlike the fixed-threshold heuristic." Future
work (`docs/limitations_and_future_work.md`) notes that a supervised
segmentation model trained on real annotated lesion masks would likely
outperform both unsupervised heuristics.

Reproduce with:
```bash
cd backend
python -m app.ml.evaluate_severity_segmentation --n_images 60
```

## 8. Fusion Strategy Comparison — Sandbox Smoke-Test Results (NOT real-world accuracy)

**Read `app/ml/compare_fusion_approaches.py`'s module docstring before
using these numbers anywhere.** This experiment was run in this
project's sandboxed development environment, which has no network
access to download either the real PlantVillage dataset or Keras'
pretrained ImageNet weights. To still exercise and validate the full
three-way comparison CODE end-to-end, the experiment below uses:
- The small, fully synthetic image benchmark (`app/ml/synthetic_benchmark.py`,
  20 images/class, class-distinguishing colored blobs, NOT real disease photos)
- A MobileNetV2 backbone with **random (non-ImageNet) initial weights**
  (`--weights none`), since ImageNet weights could not be downloaded
- Synthetically paired, class-conditional field-context values (see
  `app/ml/train_feature_fusion.py`) for the feature-level fusion model

**Actual measured results (n=60 synthetic test images, 10 classes):**

| Approach | Accuracy | Weighted F1 | Confidence calibration gap |
|---|---|---|---|
| Image-only | 0.1000 | 0.0182 | +0.0000 |
| Weighted late fusion (production default) | 0.1000 | 0.0182 | +0.0000 |
| Feature-level fusion (experimental) | 0.6333 | 0.6255 | n/a |

**How to read this correctly (important):**
- Image-only accuracy (10%) is exactly random chance for 10 classes —
  expected and correct, because the MobileNetV2 backbone has random
  (untrained) weights, not ImageNet features, in this sandboxed run.
  This is a smoke test proving the code runs, not a disease-detection
  accuracy claim.
- Late fusion's accuracy is identical to image-only **by design** (it
  only adjusts confidence, not the label — see `fusion.py`). Its
  calibration gap is 0.0000 here only because the underlying image
  model is so undertrained that its raw softmax outputs are close to
  uniform regardless of correctness, leaving nothing for the fusion
  step to calibrate against. A real, properly trained image model
  would show a non-trivial calibration gap for late fusion to adjust.
- Feature-level fusion's much higher accuracy (63.3%) is **not**
  evidence that feature-level fusion is intrinsically better. It
  reflects the fact that, in our synthetic pairing rule, field-context
  values are DELIBERATELY strongly informative about the class label
  (see `_synthesize_paired_field_context` — healthy vs. diseased
  classes are drawn from clearly separated distributions on purpose).
  With an uninformative image backbone, the joint classifier learns to
  lean almost entirely on this synthetic field-context signal. Real
  field conditions are far noisier and less directly diagnostic than
  this synthetic pairing, so this specific magnitude of improvement
  must NOT be reported as a real-world result.

**What this experiment DOES honestly establish:** all three code paths
(image-only inference, late-fusion confidence adjustment, and
feature-level joint-model training/inference) run correctly end-to-end,
train without errors, and produce internally consistent, reproducible
metrics. This is a valid "the implementation works" claim for your
report. It is NOT a valid "feature-level fusion beats late fusion by
53 points" claim — that requires re-running this unmodified script with
a real, ImageNet-pretrained MobileNetV2 trained on real PlantVillage
data, and with REAL paired field-context data if it becomes available
(see `docs/limitations_and_future_work.md`).

Reproduce with:
```bash
cd backend
python -m app.ml.synthetic_benchmark --images_per_class 20
python -m app.ml.train_transfer --data_dir ../data/synthetic_benchmark --weights none \
    --epochs 3 --fine_tune_epochs 5 --out ./storage/models/disease_model_synthtest.keras
python -m app.ml.train_feature_fusion --image_model ./storage/models/disease_model_synthtest.keras \
    --data_dir ../data/synthetic_benchmark --out ./storage/models/feature_fusion_synthtest.keras
python -m app.ml.compare_fusion_approaches \
    --image_model ./storage/models/disease_model_synthtest.keras \
    --feature_fusion_model ./storage/models/feature_fusion_synthtest.keras \
    --field_preprocessor ./storage/models/feature_fusion_synthtest_field_preprocessor.joblib \
    --data_dir ../data/synthetic_benchmark
```
To get real, report-ready numbers: download PlantVillage (`data/README.md`),
run `train_transfer.py` with the default `--weights imagenet` (requires
network access to `storage.googleapis.com`, available outside this
sandbox), then re-run `train_feature_fusion.py` and
`compare_fusion_approaches.py` pointing at the real trained model and
real data directory.
