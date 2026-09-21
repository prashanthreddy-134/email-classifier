from pathlib import Path
import re

import joblib
import pandas as pd


# =========================================================
# MODEL PATH
# =========================================================

MODEL = Path(__file__).parent / "models" / "email_classifier.joblib"


# =========================================================
# EMAIL TEXT CLEANING
# =========================================================

def clean_email(text):
    """
    Apply the same preprocessing used during model training.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize line breaks
    text = text.replace("\r", " ").replace("\n", " ")

    # Replace email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " EMAIL_ADDRESS ",
        text,
    )

    # Replace URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL_LINK ",
        text,
        flags=re.IGNORECASE,
    )

    # Replace HTML tags
    text = re.sub(
        r"<[^>]+>",
        " HTML_TAG ",
        text,
    )

    # Replace long hexadecimal IDs
    text = re.sub(
        r"\b[a-fA-F0-9]{12,}\b",
        " TRACKING_ID ",
        text,
    )

    # Replace long alphanumeric IDs
    text = re.sub(
        r"\b[A-Za-z0-9_-]{16,}\b",
        " LONG_ID ",
        text,
    )

    # Replace numbers
    text = re.sub(
        r"\b\d+(?:[.,]\d+)*\b",
        " NUMBER ",
        text,
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# EMAIL CLASSIFIER
# =========================================================

class EmailClassifier:

    def __init__(self, path=MODEL):

        if not path.exists():
            raise FileNotFoundError(
                "Model not found. Run: python train.py"
            )

        self.pipeline = joblib.load(path)

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    def predict(self, text):

        cleaned_text = clean_email(text)

        label = self.pipeline.predict(
            [cleaned_text]
        )[0]

        probabilities = self.pipeline.predict_proba(
            [cleaned_text]
        )[0]

        confidence = max(probabilities)

        return (
            str(label),
            float(confidence),
        )