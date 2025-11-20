import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure dirs exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model Configs
BERTOPIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

# Domain Configs
DOMAINS = {
    "mha": "Mental Health Apps",
    "quick_commerce": "Quick Commerce"
}

# MHA Determinants (Reference)
MHA_DETERMINANTS = [
    "Mindful meditation experience",
    "Account management",
    "Advertisement experience",
    "Personal development",
    "Tracking progress",
    "Emotional well-being",
    "UI & Design",
    "Soothing audio experience",
    "Smoking cessation support",
    "Payment & Subscription management"
]

# Quick Commerce Determinants (Reference)
QC_DETERMINANTS = [
    "Delivery speed",
    "Order accuracy",
    "Product availability",
    "Packaging quality",
    "Refund & return handling",
    "Pricing & fees",
    "App interface",
    "Customer support",
    "Rider behavior",
    "Real-time tracking"
]
