"""
Canonical class labels for the MVP.

Scope decision (documented in docs/problem_statement.md):
We intentionally support 3 crops and a manageable set of diseases so a
4-person undergraduate team can collect/curate data, train, and debug
within a semester. This list maps 1:1 onto folder names used in the
PlantVillage dataset (see data/README.md for download instructions),
so `flow_from_directory` / `image_dataset_from_directory` need no
relabeling.
"""

CLASS_NAMES = [
    "Tomato___Healthy",
    "Tomato___Early_Blight",
    "Tomato___Late_Blight",
    "Tomato___Leaf_Mold",
    "Potato___Healthy",
    "Potato___Early_Blight",
    "Potato___Late_Blight",
    "Corn___Healthy",
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
]

# Which crop each class belongs to (used to filter/validate predictions
# against the crop_type the user selected in the field-context form).
CLASS_TO_CROP = {c: c.split("___")[0].lower() for c in CLASS_NAMES}

# Human-readable display names + one-line description used in the
# explanation text returned to the frontend.
CLASS_INFO = {
    "Tomato___Healthy": "No visible disease symptoms detected on the tomato leaf.",
    "Tomato___Early_Blight": "Dark concentric ring spots typical of Alternaria solani infection.",
    "Tomato___Late_Blight": "Irregular water-soaked lesions typical of Phytophthora infestans.",
    "Tomato___Leaf_Mold": "Yellow patches on top with olive-green mold typical of Passalora fulva.",
    "Potato___Healthy": "No visible disease symptoms detected on the potato leaf.",
    "Potato___Early_Blight": "Target-like brown lesions typical of Alternaria solani.",
    "Potato___Late_Blight": "Dark, water-soaked, rapidly spreading lesions typical of Phytophthora infestans.",
    "Corn___Healthy": "No visible disease symptoms detected on the corn leaf.",
    "Corn___Common_Rust": "Small reddish-brown pustules typical of Puccinia sorghi.",
    "Corn___Gray_Leaf_Spot": "Rectangular gray-tan lesions typical of Cercospora zeae-maydis.",
}

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)  # matches MobileNetV2 default input
