# Dataset: Download, Organize, Clean, Preprocess, Split

This project uses a subset of the **PlantVillage** dataset, restricted
to 3 crops (Tomato, Potato, Corn) and 10 classes total (see
`backend/app/ml/labels.py` for the exact class list). See
`docs/methodology.md` Section 3 for the full dataset comparison and why
PlantVillage was chosen over PlantDoc for the MVP.

## 1. Download
PlantVillage is available from several public mirrors. As of writing,
a commonly used Kaggle mirror is:

```
https://www.kaggle.com/datasets/emmarex/plantdisease
```

**Verify the exact URL, license, and terms of use at download time** —
dataset mirrors change over time. Steps:
1. Create a free Kaggle account if you don't have one.
2. Install the Kaggle CLI: `pip install kaggle`
3. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json`
   (see Kaggle's API documentation for how to generate this token).
4. Download and unzip:
   ```bash
   kaggle datasets download -d emmarex/plantdisease -p data/raw --unzip
   ```
5. Confirm you now have a folder structure like:
   ```
   data/raw/PlantVillage/<ClassName>/*.jpg
   ```
   (exact nesting depends on the mirror — you may need to move the
   inner folder up one level so `data/raw/<ClassName>/` is correct).

## 2. Match Class Folder Names
`backend/app/ml/labels.py` expects these exact folder names under
`data/raw/`:
```
Tomato___Healthy
Tomato___Early_Blight
Tomato___Late_Blight
Tomato___Leaf_Mold
Potato___Healthy
Potato___Early_Blight
Potato___Late_Blight
Corn___Healthy
Corn___Common_Rust
Corn___Gray_Leaf_Spot
```
Most PlantVillage mirrors use very similar names (sometimes with
different casing/underscore placement, e.g. `Tomato_healthy` or
`Corn_(maize)___Common_rust_`). Rename the folders you need to match
exactly, and ignore/delete the other 28 classes not used by this MVP
(or simply don't copy them — `dataset_utils.py` only processes classes
it recognizes).

## 3. Clean (remove corrupted files)
Run this quick check before splitting:
```python
from PIL import Image
import pathlib

bad = []
for p in pathlib.Path("data/raw").rglob("*.*"):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        try:
            img = Image.open(p)
            img.verify()
        except Exception:
            bad.append(p)

print(f"Found {len(bad)} corrupted files")
for p in bad:
    p.unlink()  # delete corrupted files
```

## 4. Organize into train/val/test splits
From the `backend/` directory:
```bash
cd backend
python -m app.ml.dataset_utils --raw_dir ../data/raw --processed_dir ../data/processed
```
This produces:
```
data/processed/train/<ClassName>/*.jpg   (70%)
data/processed/val/<ClassName>/*.jpg     (15%)
data/processed/test/<ClassName>/*.jpg    (15%)
```
and prints a per-class count summary — paste this table into your
report's dataset description section.

## 5. Preprocessing (handled automatically at training time)
No manual preprocessing step is required beyond organizing files —
`app/ml/train_cnn.py` and `app/ml/train_transfer.py` handle resizing
(224x224), pixel scaling, and augmentation internally via
`tf.keras.utils.image_dataset_from_directory` and Keras preprocessing
layers.

## 6. (Optional) PlantDoc for realistic test-only evaluation
To honestly measure the lab-to-field generalization gap (see
`docs/literature_review.md`), download a small sample of PlantDoc
images (https://github.com/pratikkayal/PlantDoc-Dataset — verify link
at download time) for your 3 supported crops, and evaluate your trained
model on them using `app/ml/evaluate.py --data_dir <plantdoc_path>
--split test` (point `--split` at a folder containing your PlantDoc
subset organized the same way as `data/processed/test`).

## 7. Sample / Synthetic Field-Context Data
Real field-context data (temperature, humidity, rainfall, soil
moisture) paired with actual disease outbreaks is not publicly
available for this MVP. `backend/sample_data/field_context_sample.csv`
contains example rows for manual testing/demo purposes, and
`backend/app/ml/train_field_model.py` documents exactly how the
training data for the Random Forest risk model is synthetically
generated from domain rules (clearly disclosed, not hidden — see
`docs/methodology.md` Section 7).
