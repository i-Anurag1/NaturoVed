# Demo Flow & Final Presentation Structure (v2)

## 1. Live Demo Flow (rehearse this exact sequence before evaluation)

1. **Health check** — open `http://localhost:8000/docs` briefly, hit
   `GET /health`. If you trained real models beforehand,
   `image_model_loaded`/`field_model_loaded` should read `true`. If
   not, that's fine — say so directly: "the system is currently running
   in its documented fallback mode; every feature still works."
2. **Home page** — show the leaf-image upload and field-context form.
3. **Live weather demo (v2)** — click "Use Live Weather," grant location
   permission, show the form auto-fill. Then manually edit one field to
   show the fallback/override still works — this proves manual entry
   was never removed, only supplemented.
4. **Case 1 — Healthy leaf, low-risk conditions.** Upload a healthy leaf
   image with low humidity/rainfall. Show: "Healthy" status, general
   monitoring tips, low field risk.
5. **Case 2 — Diseased leaf, high-risk conditions.** Upload a diseased
   leaf with high humidity + moderate temp + recent rainfall. Show:
   disease detected, severity %, high field risk, the explanation
   describing the fusion confidence boost, AND the field-context
   explanation panel (v2) showing which environmental feature mattered most.
6. **Grad-CAM section (v2) — be careful here.** Show that the Grad-CAM
   panel renders an original-image/heatmap side-by-side. Then say
   explicitly: "this demonstrates the explainability pipeline is fully
   wired end-to-end; the heatmap itself won't align meaningfully with
   the lesion until we train on real data with ImageNet weights — see
   our limitations doc." Do not let this go unstated; a "why does the
   heatmap not highlight the lesion?" question is easy to answer
   honestly if you say it first.
7. **PDF report (v2)** — click "Download PDF Report" on the result page,
   open the downloaded file, show the image + Grad-CAM + all fields in
   the generated report.
8. **Dashboard (v2)** — navigate to the Dashboard page, show the live
   distributions updating in real time as you run more predictions.
9. **Language switch (v2)** — click the Tamil toggle, show the UI
   relabeling.
10. **Advanced options / feature-level fusion (v2, optional)** — expand
    "Advanced Options," switch to "Feature-Level Fusion (experimental),"
    resubmit the same image. Point out the "Experimental" tag on the
    result, and be ready to explain (see viva doc) why its accuracy
    numbers on synthetic data should not be over-interpreted.
11. **History page** — show all predictions logged; click into one.
12. **(If trained)** Show `docs/evaluation_plan.md`'s confusion matrix
    and accuracy numbers for the CNN baseline vs MobileNetV2.

## 2. Suggested Slide Deck Structure
1. Title — project name, team, institution, "v2" if distinguishing from
   a prior submission
2. Problem Statement (from `docs/problem_statement.md`)
3. Objectives
4. System Architecture (diagram from `docs/methodology.md`, now
   including Grad-CAM, weather, dashboard, PDF report, and the two
   fusion paths)
5. Dataset (PlantVillage subset choice, class list, split sizes)
6. Models Compared — CNN baseline vs MobileNetV2 vs (if trained) real
   fusion results
7. **v2 Feature Walkthrough** — one slide each for: Grad-CAM (with the
   explicit "not yet meaningful" caveat clearly on the slide, not just
   spoken), live weather integration, feature-level fusion (with the
   synthetic-data caveat clearly on the slide), field-context
   explainability, dashboard, PDF report, i18n
8. Severity-Segmentation Experiment — present the REJECTED v2 result as
   a genuine finding, not a failure to hide
9. Live Demo (see flow above)
10. Results Summary table
11. Limitations — pull directly from `docs/limitations_and_future_work.md`
    Sections 2 and 3 (Grad-CAM, synthetic-data caveats) — these should
    be presented confidently as evidence of rigor, not buried
12. Future Work
13. Conclusion
14. Q&A

## 3. Anticipated Evaluator Questions
See `docs/viva_questions_and_answers.md` for a full prepared set,
including the exact wording for the two hardest questions you will
likely get:
- "Why doesn't the Grad-CAM heatmap highlight the actual lesion?"
- "Doesn't your fusion comparison prove feature-level fusion is better?"

Both have honest, prepared, non-defensive answers in that document.
Read them before your viva — evaluators respond far better to "here's
exactly why, and here's exactly what would fix it" than to a vague or
overconfident answer.
