# Literature Review Guidance

This document tells you **how** to build the literature review section
of your report — it does not fabricate citations, since exact papers,
authors, and publication years must be verified by you against real
sources (Google Scholar, IEEE Xplore, ACM Digital Library, arXiv) before
they appear in your submitted report. Do not paste placeholder citations
into your final report without checking them.

## Suggested search queries
Use these on Google Scholar / IEEE Xplore to find real, citable papers:
- "PlantVillage CNN disease classification"
- "transfer learning MobileNetV2 plant disease detection"
- "leaf disease severity estimation image processing"
- "crop disease severity segmentation OpenCV"
- "multimodal plant disease detection environmental data"
- "plant disease risk prediction weather data machine learning"
- "PlantDoc dataset benchmark"
- "explainable AI plant disease diagnosis"

## What to look for and compare
For each paper you read, extract:
1. **Dataset used** (PlantVillage, PlantDoc, a custom field dataset, etc.)
2. **Model architecture** (custom CNN, VGG16, ResNet, MobileNet,
   EfficientNet, Vision Transformer, etc.)
3. **Reported accuracy/F1** — note whether it was tested on a held-out,
   real-world (field-captured) test set or only on lab-condition images
   from the same distribution as training data. This distinction matters:
   PlantVillage images are lab/lightbox photos, and models trained only
   on them tend to perform much worse on real field photos (a well
   documented generalization gap worth discussing).
4. **Whether the paper uses any structured/environmental data** — most
   will not; explicitly note this as it supports your research-gap
   argument.
5. **Whether severity/affected-area is estimated**, and how (segmentation,
   classification, regression).

## Suggested structure for your literature review section
1. Early CNN-based approaches on PlantVillage (custom/shallow CNNs)
2. Transfer-learning approaches (VGG/ResNet/MobileNet/EfficientNet
   fine-tuned on PlantVillage or PlantDoc)
3. Severity/segmentation-focused works (lesion area estimation)
4. Any works combining environmental/weather data with disease models
   (likely sparse — supports your novelty claim)
5. Summary table comparing dataset, model, accuracy, and whether
   severity/context was addressed
6. Explicit statement of the gap your project addresses (link back to
   `docs/problem_statement.md` Section 5)

## Academic honesty note
Do not state performance numbers for other papers without verifying
them against the actual publication. Do not claim your project
"outperforms" cited works unless you have run a fair, documented
comparison under comparable conditions — for an undergraduate MVP, it
is more honest and more defensible in a viva to say your accuracy is
"in a comparable range to lab-condition PlantVillage baselines
reported in the literature" than to claim superiority.
