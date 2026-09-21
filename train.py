from pathlib import Path
import re

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# =========================================================
# PATHS
# =========================================================

BASE = Path(__file__).parent

DATA_FILE = BASE / "data" / "emails.csv"
MODEL_DIR = BASE / "models"
MODEL_FILE = MODEL_DIR / "email_classifier.joblib"


# =========================================================
# EMAIL TEXT CLEANING
# =========================================================

def clean_email(text):
    """
    Normalize email text while preserving useful language patterns.
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
# LEGITIMATE REAL-WORLD EMAIL EXAMPLES
# =========================================================

LEGITIMATE_EXAMPLES = [

    # ---------------- ORDER / SHOPPING ----------------

    """
    Subject: Your order has been confirmed

    Thank you for your order. Your purchase has been successfully confirmed.
    You can view your order details and expected delivery date from your account.
    Order number: ORDER_ID
    """,

    """
    Subject: Your order has shipped

    Your order has been shipped and is on its way.
    You can track the package using the tracking link in your account.
    Thank you for shopping with us.
    """,

    """
    Subject: Your package was delivered

    Your package was delivered today.
    The package was handed to the delivery recipient.
    You can view the delivery details and order information in your account.
    """,

    """
    Subject: Delivery update for your order

    We wanted to let you know that your order is scheduled for delivery.
    You can check the latest delivery status from your order history.
    """,

    """
    Subject: Your purchase receipt

    Thank you for your purchase.
    This email confirms your transaction and contains your receipt.
    You can view the complete order information in your account.
    """,

    """
    Subject: Order cancellation confirmed

    Your order cancellation has been successfully processed.
    If a refund is applicable, the amount will be returned according to the
    payment method used for the purchase.
    """,

    """
    Subject: Refund processed

    Your refund has been processed successfully.
    The refunded amount will appear in your original payment method.
    You can review the transaction details in your account.
    """,

    # ---------------- PAYMENT ----------------

    """
    Subject: Payment received

    We have successfully received your payment.
    Transaction reference: TRANSACTION_ID
    You can review your payment history from your account.
    """,

    """
    Subject: Payment confirmation

    Your payment was successfully completed.
    Thank you for using our service.
    Please keep this confirmation for your records.
    """,

    """
    Subject: Invoice available

    Your latest invoice is now available in your account.
    You can view or download the invoice from the billing section.
    """,

    """
    Subject: Subscription payment confirmation

    Your subscription payment has been successfully processed.
    Your subscription remains active and no action is required.
    """,

    # ---------------- ACCOUNT ----------------

    """
    Subject: Your account verification

    Please verify your email address to complete the account registration.
    If you created this account, follow the verification instructions.
    """,

    """
    Subject: Password reset request

    We received a request to reset the password for your account.
    If you made this request, follow the password reset instructions.
    If you did not request a password reset, you can ignore this message.
    """,

    """
    Subject: Security notification

    A security change was recently made to your account.
    If you recognize this activity, no further action is required.
    """,

    """
    Subject: New sign-in to your account

    Your account was recently accessed from a new device.
    If this was you, you can safely ignore this notification.
    """,

    # ---------------- DELIVERY / SERVICE ----------------

    """
    Subject: Delivery scheduled

    Your delivery has been scheduled.
    You can check the delivery window and tracking information from your account.
    """,

    """
    Subject: Package tracking update

    Your shipment status has been updated.
    The package is currently in transit.
    Please use the tracking page in your account for the latest information.
    """,

    """
    Subject: Service appointment confirmed

    Your appointment has been confirmed.
    Date and time details are available in your account.
    Please contact support if you need to reschedule.
    """,

    # ---------------- BANKING / FINANCE ----------------

    """
    Subject: Transaction notification

    A transaction was successfully processed on your account.
    You can review the transaction details through your secure banking account.
    """,

    """
    Subject: Monthly account statement

    Your monthly account statement is now available.
    Please sign in to your account to review the statement.
    """,

    """
    Subject: Card payment notification

    A card payment has been successfully processed.
    Please review your account activity if you do not recognize the transaction.
    """,

    # ---------------- WORK / BUSINESS ----------------

    """
    Subject: Meeting reminder

    This is a reminder about the project meeting scheduled for tomorrow.
    Please review the agenda and bring your latest project updates.
    """,

    """
    Subject: Project status update

    The latest project status report is now available.
    Please review the assigned tasks and current deadlines.
    """,

    """
    Subject: Document shared with you

    A document has been shared with you.
    You can access it through the organization's document system.
    """,

    """
    Subject: Task assignment

    A new task has been assigned to you.
    Please review the task details and complete it before the stated deadline.
    """,

    # ---------------- JOB / EDUCATION ----------------

    """
    Subject: Application received

    Thank you for applying.
    We have successfully received your application and will review it shortly.
    """,

    """
    Subject: Interview confirmation

    Your interview has been scheduled.
    Please review the date, time, and meeting details in your application portal.
    """,

    """
    Subject: Course enrollment confirmation

    Your enrollment has been successfully completed.
    Course information and schedule details are available in your student account.
    """,

    """
    Subject: Assignment submission confirmation

    Your assignment was successfully submitted.
    You can view the submission status from your student portal.
    """,

    # ---------------- GENERAL LEGITIMATE ----------------

    """
    Subject: Thank you for contacting support

    We received your support request.
    Our team will review your message and respond as soon as possible.
    Your support request has been recorded in our system.
    """,

    """
    Subject: Your requested information

    The information you requested is now available.
    Please sign in to your account to view the details.
    """,

    """
    Subject: Service notification

    This is an automated notification regarding your account.
    No action is required unless stated in the message.
    """,

    """
    Subject: Confirmation

    This email confirms that your request has been successfully completed.
    You can review the details in your account.
    """,

    """
    Subject: Account notification

    This is a notification regarding recent activity on your account.
    Please review your account if you have any questions.
    """,
]


# =========================================================
# CREATE VARIATIONS
# =========================================================

def create_legitimate_variations(examples, copies=6):
    """
    Create small natural variations of legitimate examples.
    This helps the model learn patterns rather than one exact sentence.
    """

    variations = []

    prefixes = [
        "",
        "Hello, ",
        "Hi, ",
        "Dear Customer, ",
        "Dear User, ",
    ]

    closings = [
        "",
        " Thank you.",
        " Regards.",
        " Best regards.",
        " Please keep this message for your records.",
    ]

    for example in examples:

        for i in range(copies):

            prefix = prefixes[i % len(prefixes)]
            closing = closings[i % len(closings)]

            variation = example.strip()

            if prefix:
                variation = prefix + variation

            variation += closing

            variations.append(variation)

    return variations


# =========================================================
# LOAD ORIGINAL DATASET
# =========================================================

print("=" * 60)
print("AI EMAIL CLASSIFIER - IMPROVED MODEL TRAINING")
print("=" * 60)

print("\nLoading original dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = {"text", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "Dataset must contain 'text' and 'label' columns."
    )

df = df.dropna(subset=["text", "label"]).copy()

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert ham to legitimate
df["label"] = df["label"].replace(
    {
        "ham": "legitimate"
    }
)

# Keep only valid classes
df = df[
    df["label"].isin(
        [
            "spam",
            "legitimate",
        ]
    )
].copy()

# Clean original dataset
df["clean_text"] = df["text"].apply(clean_email)

# Remove empty messages
df = df[
    df["clean_text"].str.len() > 0
].copy()


print(f"Original emails: {len(df):,}")
print(
    f"Original legitimate: "
    f"{(df['label'] == 'legitimate').sum():,}"
)
print(
    f"Original spam: "
    f"{(df['label'] == 'spam').sum():,}"
)


# =========================================================
# SPLIT ORIGINAL DATASET FIRST
# =========================================================

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# =========================================================
# ADD LEGITIMATE TRAINING DATA
# =========================================================

print("\nAdding realistic legitimate email examples...")

additional_legitimate = create_legitimate_variations(
    LEGITIMATE_EXAMPLES,
    copies=6,
)

additional_legitimate = [
    clean_email(text)
    for text in additional_legitimate
]

additional_labels = [
    "legitimate"
] * len(additional_legitimate)

extra_df = pd.DataFrame(
    {
        "text": additional_legitimate,
        "label": additional_labels,
    }
)

# Add ONLY to training data.
X_train = pd.concat(
    [
        X_train.reset_index(drop=True),
        extra_df["text"].reset_index(drop=True),
    ],
    ignore_index=True,
)

y_train = pd.concat(
    [
        y_train.reset_index(drop=True),
        extra_df["label"].reset_index(drop=True),
    ],
    ignore_index=True,
)


print(
    f"Added legitimate training examples: "
    f"{len(additional_legitimate):,}"
)

print(
    f"Final training emails: "
    f"{len(X_train):,}"
)

print(
    f"Final test emails: "
    f"{len(X_test):,}"
)


# =========================================================
# HYBRID TF-IDF FEATURES
# =========================================================

print("\nBuilding hybrid TF-IDF model...")

word_features = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    max_features=150000,
)

character_features = TfidfVectorizer(
    analyzer="char",
    lowercase=True,
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=100000,
)


model = Pipeline(
    [
        (
            "features",
            FeatureUnion(
                [
                    (
                        "word_tfidf",
                        word_features,
                    ),
                    (
                        "char_tfidf",
                        character_features,
                    ),
                ]
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                C=2.0,
                class_weight="balanced",
            ),
        ),
    ]
)


# =========================================================
# TRAIN
# =========================================================

print("\nTraining improved model...")

model.fit(
    X_train,
    y_train,
)

print("Training completed.")


# =========================================================
# EVALUATE ON ORIGINAL HELD-OUT TEST DATA
# =========================================================

print("\nEvaluating on original held-out test data...")

predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    pos_label="spam",
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    pos_label="spam",
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="spam",
    zero_division=0,
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 60)
print("IMPROVED MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4%}")
print(f"Precision: {precision:.4%}")
print(f"Recall   : {recall:.4%}")
print(f"F1 Score : {f1:.4%}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0,
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

MODEL_DIR.mkdir(
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE,
)

print("=" * 60)

print(
    "\nModel saved to:"
)

print(
    MODEL_FILE
)

print("=" * 60)

print(
    "\nTraining finished successfully."
)