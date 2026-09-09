# Limitations, Ethical/Practical Considerations & Future Work (v2)

**Read this document before citing any v2 result in a report or demo.**
It has been substantially rewritten from v1: several v1 "future work"
items (Grad-CAM, live weather, feature-level fusion, multilingual UI)
are now built — but building them surfaced new, specific limitations
that must be disclosed with equal prominence. Nothing below is hidden
or softened; several items are flagged CRITICAL because they directly
affect how you should — and should not — present this project's results.

## 1. Core Limitations (carried over from v1, still fully applicable)
1. **Lab-condition training data.** PlantVillage images are captured
   under controlled/lightbox conditions. Real farm photos will likely
   reduce accuracy — a well-known generalization gap (see
   `docs/literature_review.md`).
2. **Unsupervised severity estimation.** The OpenCV color-segmentation
   severity module has no ground-truth lesion masks to validate
   against. Section 7 of `docs/evaluation_plan.md` documents a real
   experiment testing a candidate improvement, which was REJECTED —
   the original heuristic remains in production.
3. **Narrow crop/disease coverage.** Only 3 crops and 10 classes.
4. **Confidence, not certainty.** No model output here is a guarantee
   of correctness; this remains a decision-support tool.

## 2. CRITICAL v2 Limitation: Grad-CAM Is Not Yet Meaningful

Grad-CAM is correctly implemented and functions end-to-end (see
`backend/app/services/gradcam.py` and
`backend/tests/test_v2_features.py`), but its output is NOT currently
diagnostically meaningful, and must not be presented as such.

Why: Grad-CAM's heatmap shows which pixels most influenced the model's
prediction. That is only informative if the model has learned real
disease-relevant features. In this project's sandboxed development
environment, `storage.googleapis.com` (which hosts Keras' pretrained
ImageNet weights) is not reachable — outside the environment's network
allowlist — and neither is Kaggle (for downloading PlantVillage). As a
direct consequence, every disease model trained and tested during this
project's development used randomly initialized weights, with no
ImageNet pretraining and no real leaf photos, only the small synthetic
benchmark (`app/ml/synthetic_benchmark.py`).

We verified this concretely: generating a Grad-CAM heatmap for a
synthetic test image with known, exactly-placed lesion blobs, the
resulting heatmap's highlighted regions did NOT spatially align with
the actual lesion blobs. This is the expected, correct behavior of
Grad-CAM applied to an untrained/randomly-initialized network — it is
not a bug in the Grad-CAM implementation itself.

**What this means for you:**
- Do NOT show the current Grad-CAM output in a viva/demo as evidence
  the model "looks at the lesion." It does not, yet.
- DO demonstrate that the feature exists, runs without error, produces
  a correctly-shaped and correctly-encoded heatmap overlay, and is
  wired end-to-end through the API and frontend — this is a legitimate
  "the explainability infrastructure is built and tested" claim.
- Grad-CAM WILL become meaningful automatically, with no code changes,
  once you train the image model on real PlantVillage data with real
  ImageNet weights (`python -m app.ml.train_transfer` with the default
  `--weights imagenet`, run somewhere with normal internet access).
  Re-run the Grad-CAM visual check after that training and confirm
  heatmaps concentrate on visible lesion areas before using them in
  your final report or demo.

## 3. CRITICAL v2 Limitation: Do Not Overstate Synthetic Field-Context Results

Two separate v2 experiments use synthetic field-context data, and their
numbers must be quoted with the exact caveats below — not as
general-purpose accuracy claims:

**(a) Field-risk Random Forest (~94-97% accuracy on its own synthetic
test split).** This measures how well the model learned the
hand-written domain-rule labeling function (see
`app/ml/train_field_model.py`), not real-world epidemiological
validity. No public dataset links real field conditions to confirmed
disease outbreaks, so this could not be avoided within this project's
scope — but the accuracy number must always be presented alongside
this caveat, never standalone.

**(b) Feature-level fusion vs. image-only/late-fusion comparison
(`docs/evaluation_plan.md` Section 8).** The measured 63.3% vs 10%
accuracy gap in favor of feature-level fusion is NOT evidence that
feature-level fusion is intrinsically superior. It reflects that the
synthetic field-context pairing used for training was deliberately
constructed so that field values strongly determine the class label
(healthy vs. diseased crops draw from clearly separated synthetic
distributions). With an image backbone that has learned nothing (random
weights, see Section 2 above), the joint model simply learns to lean on
this artificially strong field-context signal. Real field conditions
are far noisier and less directly diagnostic. Re-running this
comparison after real image-model training, and ideally with real (not
synthetic) paired field-context data, is required before this
comparison can support any claim in an academic report beyond "the
three-way comparison code runs correctly end-to-end."

## 4. Weather Integration — Testing Limitation (disclosed, not hidden)
`GET /weather` calls the free Open-Meteo API (`api.open-meteo.com`),
which is also outside this sandboxed environment's network allowlist.
The integration is fully implemented and unit-tested against a mocked
HTTP response (see `test_weather_service_parses_mocked_response` in
`backend/tests/test_v2_features.py`), which verifies the
request-building and response-parsing logic is correct, but a live
call to the real API has not been exercised end-to-end in this
environment. Test this yourself with real network access before
relying on it for a live demo — see `docs/deployment.md`.

## 5. PDF Report & Dashboard — Verified, No Special Caveats
Unlike the items above, PDF report generation and the analytics
dashboard were fully exercised end-to-end in this environment with
real HTTP requests and produced verified, correct output (a valid
multi-page PDF; correct aggregate counts against a live SQLite
database). These do not carry the same "not yet meaningful" caveat.

## 6. Ethical & Practical Considerations (unchanged from v1, still apply)
- No specific chemical/fungicide dosing advice is given.
- A wrong "Healthy" prediction (false negative) could delay real
  response — this tool is decision support, not certified diagnosis.
- Data privacy: images/data stored locally only in this MVP.
- Equity of access: smartphone/internet dependency is a real
  deployment limitation.
- New for v2: live weather integration and Grad-CAM regeneration both
  add external dependencies (a third-party API call; TensorFlow
  inference on every request) — consider rate-limiting `/weather` and
  caching Grad-CAM output if deploying this beyond a classroom demo.

## 7. Future Work (genuinely not built, in priority order)
1. **Train on real PlantVillage data with real ImageNet weights** —
   this single step resolves the Grad-CAM meaningfulness gap (Section 2)
   and gives real, citable disease-classification accuracy.
2. **Real paired field-context data** (partnering with an agricultural
   extension office, or a field trial) to validate the field-risk model
   and re-run the fusion comparison honestly (Section 3).
3. **Semantic segmentation model** for severity, trained on annotated
   lesion masks.
4. **SHAP/LIME for per-prediction (local) field-context explanations**,
   upgrading from the current global feature-importance approach.
5. **GPS/location-based historical outbreak risk** (beyond current-
   moment weather).
6. **Mobile app / offline inference** (TensorFlow Lite export).
7. **IoT sensor integration** for automated soil-moisture/humidity input.
8. **Voice input** for low-literacy accessibility.
9. **Additional languages** beyond English/Tamil.
10. **Managed database (Postgres)** for durable cloud deployment
    (see `docs/deployment.md` Section 3).

Each item above was deliberately deferred — either out of MVP scope
for a semester project, or because this sandboxed development
environment's network restrictions made it impossible to build
honestly within this session (Items 1, 2 specifically).
