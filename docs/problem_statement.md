# Problem Statement, Objectives, Scope & Novelty

## 1. Problem Statement
Crop diseases cause major yield losses worldwide, and smallholder farmers
often lack timely access to plant-pathology expertise. Most existing
AI-based crop-disease tools (including the majority of PlantVillage-based
research projects) perform **image-only classification**: given a leaf
photo, they output a disease label and stop there. This ignores three
things a real agronomist would consider:

1. **How bad is it right now?** (severity / affected leaf area)
2. **Do current field conditions make the situation likely to worsen?**
   (temperature, humidity, rainfall, soil moisture, growth stage)
3. **What should the farmer actually do next?** (context-aware action,
   not just a label)

This project builds a system that answers all three questions together,
using both the leaf image and structured field-context data.

## 2. Objectives
1. Detect crop disease from a leaf image for a defined set of
   crops/diseases with acceptable accuracy on held-out data.
2. Estimate disease **severity** (Mild / Moderate / Severe) and, where
   feasible, the affected leaf-area percentage.
3. Process **structured field-context data** (temperature, humidity,
   rainfall, soil moisture, crop, growth stage) with a genuine trained
   model, not hardcoded rules only.
4. **Fuse** image-derived and field-context-derived signals into a single,
   more informative prediction than either modality alone.
5. Produce a **field/environmental risk level** describing how favorable
   current conditions are for disease development or spread.
6. Generate **deterministic, explainable, rule-based recommendations**
   grounded in the disease, its severity, and field risk — without using
   a generative model for the core decision logic.
7. Package all of the above into a working, demonstrable, full-stack web
   application (React + FastAPI + SQLite) suitable for a college
   evaluation and a public GitHub repository.

## 3. Scope (MVP boundaries — read this before promising more in a demo)
**In scope for the MVP:**
- 3 crops: Tomato, Potato, Corn (Maize)
- 10 classes total (Healthy + disease classes per crop — see
  `backend/app/ml/labels.py`)
- Severity via an OpenCV color-segmentation heuristic (unsupervised —
  no pixel-level ground truth exists in PlantVillage)
- Field-risk via a Random Forest trained on **domain-rule-generated
  synthetic data** (clearly documented as synthetic — see
  `docs/methodology.md`)
- Weighted late fusion (not feature-level fusion — see justification in
  `backend/app/services/fusion.py`)
- Local web app: React frontend + FastAPI backend + SQLite database

**Explicitly out of scope for the MVP** (see `docs/future_work.md`):
- Live weather API integration
- GPS/location-based automatic risk prediction
- Semantic segmentation with pixel-level lesion ground truth
- Grad-CAM / saliency-map explainability
- Multilingual support, mobile app, IoT sensors, voice input, offline
  inference
- More than 3 crops / broader disease coverage
- Region-specific or brand-specific pesticide/fungicide dosing advice

## 4. Novelty
The novelty of this project is **not** "yet another CNN on
PlantVillage." It is the integration of three signal types that are
normally handled by separate, disconnected tools:

1. **Visual disease symptoms** (image classification)
2. **Disease severity** (quantitative, not just categorical disease ID)
3. **Structured field/environmental conditions** (a genuinely trained
   structured-data model, fused with the image model's output)

into a single **field-context-aware risk assessment and recommendation
pipeline**, rather than a bare disease label. We are careful not to
overclaim: the field-context "risk" model is trained on synthetic,
domain-rule-labeled data (there is no public labeled dataset linking
field conditions to disease-outbreak ground truth), and this is
disclosed prominently rather than hidden — see `docs/methodology.md`
Section 5 and `docs/limitations.md`.

## 5. Research Gap
Reviewing typical PlantVillage/PlantDoc-based literature (see
`docs/literature_review.md` for guidance on what papers to read and
cite), the vast majority of published work reports image-classification
accuracy only. Few, if any, publicly available student-scale projects
combine image-based diagnosis with a trained structured-data model for
environmental risk **and** provide affected-area severity estimation
**and** a deterministic recommendation layer — end-to-end, in one
working system. That combination, implemented honestly (including
disclosing where synthetic data was used), is the gap this project
targets.
