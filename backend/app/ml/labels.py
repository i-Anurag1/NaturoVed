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
    "Pepper___Healthy",
    "Pepper___Bacterial_Spot",
    "Squash___Powdery_Mildew",
    "Broccoli___Healthy",
    "Broccoli___Alternaria_Leaf_Spot",
    "Broccoli___Black_Rot",
    "Cabbage___Healthy",
    "Cabbage___Alternaria_Leaf_Spot",
    "Cabbage___Grey_Mould",
    "Cabbage___Tip_Burn",
    "Cabbage___Damaged_Leaf",
    "Cauliflower___Healthy",
    "Cauliflower___Alternaria_Leaf_Spot",
    "Cauliflower___Bacterial_Soft_Rot",
    "Cauliflower___Yellow_Virus",
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
    "Pepper___Healthy": "No visible disease symptoms detected on the pepper leaf.",
    "Pepper___Bacterial_Spot": "Small dark, water-soaked spots that may turn brown with yellow halos, typical of Xanthomonas infection.",
    "Squash___Powdery_Mildew": "White powdery fungal growth on the leaf surface.",
    "Broccoli___Healthy": "No visible disease symptoms detected on the broccoli leaf.",
    "Broccoli___Alternaria_Leaf_Spot": "Dark, often concentric spots typical of Alternaria infection.",
    "Broccoli___Black_Rot": "V-shaped yellow lesions from the leaf edge, typical of bacterial black rot.",
    "Cabbage___Healthy": "No visible disease symptoms detected on the cabbage leaf.",
    "Cabbage___Alternaria_Leaf_Spot": "Dark, often concentric spots typical of Alternaria infection.",
    "Cabbage___Grey_Mould": "Grey fuzzy mould growth typical of Botrytis infection.",
    "Cabbage___Tip_Burn": "Browning and drying at leaf margins, usually a physiological disorder.",
    "Cabbage___Damaged_Leaf": "Physical or pest damage to the leaf, not necessarily a disease.",
    "Cauliflower___Healthy": "No visible disease symptoms detected on the cauliflower leaf.",
    "Cauliflower___Alternaria_Leaf_Spot": "Dark, often concentric spots typical of Alternaria infection.",
    "Cauliflower___Bacterial_Soft_Rot": "Water-soaked, soft, decaying tissue typical of bacterial soft rot.",
    "Cauliflower___Yellow_Virus": "Yellowing and mottling typical of a viral infection.",
}

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)  # matches MobileNetV2 default input
