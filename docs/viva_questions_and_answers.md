# Viva / Evaluation Questions & Answers (v2)

Prepare these in your own words — do not read verbatim in a viva. Each
answer points to the exact file/doc backing it so you can pull up
evidence if pressed.

## General / Architecture

**Q: Walk me through what happens when a user submits a diagnosis.**
A: Image + field-context form -> FastAPI validates and saves the image
-> field-context Random Forest assesses environmental risk -> disease
prediction runs (production: MobileNetV2 + weighted late fusion;
optional experimental path: feature-level fusion) -> OpenCV severity
module estimates affected leaf area -> explanation and rule-based
recommendations are generated -> Grad-CAM heatmap is generated if a
trained model produced the prediction -> everything is persisted to
SQLite -> the full result returns to React. See docs/methodology.md
for the diagram and backend/app/routers/predict.py for the exact code.

**Q: What changed between v1 and v2?**
A: v1 was image + field-context + late fusion + severity + rule-based
recommendations, in a working full-stack app. v2 adds: Grad-CAM
explainability, live weather auto-fill, an experimental feature-level
fusion model (compared against the v1 late-fusion baseline), field-risk
model explainability (feature importance), an analytics dashboard, PDF
report generation, English/Tamil i18n, Docker + CI, and a leaf-
segmentation improvement experiment (which was tested and rejected —
see below). Every v1 endpoint/field still works unchanged; all v2
additions are backward-compatible defaults.

## Grad-CAM

**Q: Show me the Grad-CAM heatmap and explain what it means.**
A: The Grad-CAM code is fully implemented, tested, and produces a
correctly-shaped heatmap overlay end-to-end. However, in this
project's development environment we could not download ImageNet
pretrained weights (network restrictions — see
docs/limitations_and_future_work.md Section 2), so every trained model
used for testing has random, untrained weights. We verified that the
resulting heatmaps do NOT align with actual lesion locations on a
synthetic test image with known lesion positions — this is expected
behavior for an untrained network, not a Grad-CAM bug. Once the model
is trained on real PlantVillage data with real ImageNet weights (one
command, documented in the README), Grad-CAM will automatically start
producing meaningful heatmaps with zero code changes.

**Q: Why does Grad-CAM need special handling for a nested MobileNetV2
sub-model?**
A: Because build_transfer_model() calls the MobileNetV2 base model as
a layer within a larger functional model, its internal conv layers get
a new graph "node" for that specific call. gradcam.py builds a
grad-model by extracting base_model.input directly and rebuilding the
forward pass (GAP -> Dense -> Dense) manually to avoid the ambiguous
node-index issue with nested Keras Functional models.

## Weather Integration

**Q: What happens if the weather API is down or the user denies
location permission?**
A: The frontend's WeatherFetchButton shows an inline error and the
form falls back to whatever values were already there — manual entry
was never disabled or hidden, it's the permanent fallback path, not
just an error state. Backend-side, GET /weather returns HTTP 503 with
a clear message rather than crashing, and /predict never requires a
successful weather fetch — weather_source simply gets recorded as
"manual".

**Q: Did you test the live weather API?**
A: The parsing/request logic is unit-tested against a mocked Open-Meteo
response (see test_weather_service_parses_mocked_response). A live
call was not exercised in this sandboxed development environment
because api.open-meteo.com is outside its network allowlist — this is
disclosed in docs/limitations_and_future_work.md Section 4, and should
be verified with real network access before a live demo.

## Multimodal Fusion

**Q: What's the difference between your two fusion methods?**
A: Weighted late fusion (production default) runs the image model and
field-context model independently, then adjusts the DISPLAYED
CONFIDENCE based on whether field conditions are consistent with the
diagnosis — it never changes the predicted disease label. Feature-level
fusion (experimental) is a separately trained joint neural network that
takes the image embedding concatenated with processed field features
as input to a single classifier — this CAN change the predicted label,
not just the confidence.

**Q: Your comparison shows feature fusion massively outperforming
image-only. Doesn't that prove feature-level fusion is better?**
A: No — and I want to be upfront about that. In our sandboxed
environment, the image model has random (untrained) weights, so
image-only accuracy is exactly chance level. Feature-level fusion's
higher accuracy comes almost entirely from the synthetic field-context
pairing we generated, which was deliberately constructed to strongly
correlate with the class label. Real field conditions are much noisier.
This experiment proves the code works correctly end-to-end; it does
not prove feature-level fusion is superior in the real world. See
docs/evaluation_plan.md Section 8 and docs/limitations_and_future_work.md
Section 3 for the full, explicit caveat.

**Q: Why keep late fusion as the default if you built feature fusion?**
A: Late fusion doesn't require a jointly-labeled dataset to train
(which doesn't exist publicly for this problem), is easier to explain
and audit (an explicit formula, not a black-box joint network), and
degrades gracefully if either sub-model is retrained independently.
Feature-level fusion is offered as an opt-in experimental alternative
for transparency and comparison, not because it's proven better yet.

## Field-Context Explainability

**Q: How do you explain the field-risk prediction?**
A: We use the trained Random Forest's built-in feature_importances_
(global, model-wide importance), then combine that with the specific
values submitted in a request to produce a readable, per-request
explanation (e.g., "Humidity (82%) was the most influential factor —
model-wide importance 34%"). We chose this over SHAP/LIME because it
requires no extra dependency, has no extra inference latency, and is
straightforward to defend in a viva. SHAP-based local explanations are
listed as future work.

## Severity / Leaf Segmentation

**Q: Did you try to improve the severity estimation method?**
A: Yes — and this is a good example of an honest negative result. We
built a synthetic benchmark with exactly known ground-truth affected-
area percentages, then compared our production HSV-threshold heuristic
against a candidate LAB-color-space + Otsu-thresholding method. The
candidate looked promising in theory (Otsu self-calibrates per image),
but it catastrophically failed on healthy leaves — Otsu always finds
some threshold split even when there's no lesion, so it flagged ~50%
of every healthy leaf as diseased. We kept the original method and
documented the rejected experiment in docs/evaluation_plan.md Section 7.

## Dashboard / PDF Report

**Q: Is the analytics dashboard live data or mocked?**
A: Live — it runs real SQL aggregate queries (GROUP BY counts) against
the actual SQLite predictions table via SQLAlchemy. We verified this
end-to-end: submitted real predictions via /predict, then confirmed
/dashboard/summary reflected the exact updated counts.

**Q: What's in the PDF report and how is it generated?**
A: Generated with ReportLab (pure Python, no external binary
dependency). Contains the original image, Grad-CAM overlay (when
available), predicted disease, both confidence scores, severity and
affected area, all submitted field conditions, field risk, the full
explanation text, and all recommendations, plus a disclaimer. We
verified it produces a valid multi-page PDF (checked the %PDF magic
bytes and page count).

## i18n

**Q: Why English and Tamil specifically, and why not a full i18n
library?**
A: Tamil was specified in the project brief; a lightweight custom
context + dictionary (rather than pulling in react-i18next or similar)
keeps the bundle small and the translation logic transparent for a
4-person team to extend to more languages later — adding a language is
just adding one more key-value object to
frontend/src/i18n/translations.js.

## Docker / CI / Deployment

**Q: Did you actually run Docker to confirm it works?**
A: The docker-compose.yml and both Dockerfiles were validated
statically in this project's development environment (YAML syntax,
every environment variable cross-checked one-by-one against
app/config.py, all referenced file paths confirmed to exist) — but
this sandboxed environment does not have Docker installed, so the
actual docker compose up --build command has not been run here. This
is disclosed explicitly in docs/deployment.md — run it yourself before
a live demo and confirm both containers report healthy.

**Q: Does your CI actually pass?**
A: Every command in .github/workflows/ci.yml was run manually, in this
environment, with the same working directory and arguments the
workflow specifies, and all succeeded (pytest: 30/30 passing; frontend
npm run build: succeeded; the ML smoke-test steps: all completed
without error). The workflow itself has not been executed on actual
GitHub Actions infrastructure, since no GitHub repository was created
during this session — push this repo to GitHub and confirm the Actions
tab shows green.

## Honesty / Methodology (expect this question)

**Q: A lot of your numbers come with big caveats. Why not just train on
real data and get clean results?**
A: This project was built in a sandboxed development environment with
a restricted network allowlist that blocks both Kaggle (PlantVillage
download) and Google's model-weight hosting (ImageNet weights for
MobileNetV2). Every piece of code needed to do real training is
written, documented, and was validated on synthetic data as a
correctness smoke test — but the actual PlantVillage training run
needs to happen in an environment with normal internet access, which
is a one-command process documented in the README. We chose to be
explicit about which numbers are "code works correctly" versus "this
is a validated real-world result," rather than blur that distinction.
