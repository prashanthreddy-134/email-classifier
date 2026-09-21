import textwrap
import streamlit as st
from classifier import EmailClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Email Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(92, 65, 255, 0.14), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(0, 170, 255, 0.10), transparent 30%),
        linear-gradient(135deg, #080b12 0%, #0b0f18 50%, #070a10 100%);
    color: #f5f7fb;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080b12 0%, #0b0f17 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff;
}

.hero {
    padding: 34px 38px;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(83, 63, 170, 0.20),
            rgba(24, 123, 170, 0.15)
        );
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow:
        0 20px 60px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #ffffff;
}

.hero-title span {
    background: linear-gradient(90deg, #8b5cf6, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 16px;
    line-height: 1.7;
    color: #c9d0dc;
    max-width: 900px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 22px;
    font-weight: 750;
    color: #ffffff;
    margin: 10px 0 12px 0;
}

.metric-card {
    background: rgba(18, 22, 32, 0.82);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 20px;
    min-height: 112px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8994a5;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
}

.result-card {
    border-radius: 22px;
    padding: 28px;
    margin-top: 22px;
    background: rgba(17, 21, 30, 0.92);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 18px 45px rgba(0,0,0,0.25);
}

.result-spam {
    border: 1px solid rgba(255, 75, 75, 0.45);
    background:
        linear-gradient(
            135deg,
            rgba(255, 70, 70, 0.13),
            rgba(25, 25, 35, 0.92)
        );
}

.result-legitimate {
    border: 1px solid rgba(46, 204, 113, 0.40);
    background:
        linear-gradient(
            135deg,
            rgba(46, 204, 113, 0.11),
            rgba(25, 25, 35, 0.92)
        );
}

.result-label {
    font-size: 30px;
    font-weight: 850;
    margin-bottom: 8px;
    color: #ffffff;
}

.result-confidence {
    font-size: 17px;
    color: #cbd3df;
}

.result-note {
    margin-top: 18px;
    padding: 15px 18px;
    border-radius: 13px;
    background: rgba(255,255,255,0.045);
    color: #bfc8d6;
    font-size: 14px;
    line-height: 1.6;
}

.info-card {
    padding: 20px 22px;
    border-radius: 17px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    line-height: 1.7;
    color: #c8d0dc;
}

.footer {
    text-align: center;
    color: #687385;
    font-size: 13px;
    padding-top: 30px;
}

textarea {
    border-radius: 15px !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_classifier():
    return EmailClassifier()


try:
    clf = load_classifier()

except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

if "result" not in st.session_state:
    st.session_state.result = None


# =========================================================
# EXAMPLE FUNCTIONS
# =========================================================

def set_spam_example():
    st.session_state.email_text = """URGENT: You have been selected for a special cash reward!

Congratulations! Your email address has been randomly selected to receive a $5,000 reward.

To claim your prize, confirm your account immediately by clicking the verification link below.

This offer expires today. Act now to avoid losing your reward.

Claim your reward now and receive your money instantly.

Congratulations once again!"""

    st.session_state.result = None


def set_legitimate_example():
    st.session_state.email_text = """Subject: Project Meeting – Tomorrow at 10:00 AM

Hi Team,

Just a reminder that we have our project meeting tomorrow at 10:00 AM.

We will review the current development progress, discuss the remaining tasks, and finalize the timeline for the next phase.

Please bring your latest updates and any questions you would like to discuss.

Regards,
Project Team"""

    st.session_state.result = None


def clear_email():
    st.session_state.email_text = ""
    st.session_state.result = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ Model Information")

    st.markdown("### Algorithm")
    st.write("Hybrid TF-IDF + Logistic Regression")

    st.markdown("### Training Dataset")
    st.write("5,642 original emails")

    st.markdown("### Test Accuracy")
    st.write("98.87%")

    st.markdown("### Spam Precision")
    st.write("99.77%")

    st.markdown("### Spam Recall")
    st.write("96.63%")

    st.markdown("### Spam F1 Score")
    st.write("98.17%")

    st.divider()

    st.markdown("## 📊 Dataset")

    st.write("📨 **Total emails:** 5,642")
    st.write("🟢 **Legitimate:** 3,861")
    st.write("🔴 **Spam:** 1,781")

    st.divider()

    st.markdown("## 🧠 How It Works")

    st.write(
        "The email is cleaned and transformed using both "
        "word-level and character-level TF-IDF features. "
        "Logistic Regression then classifies the message "
        "as spam or legitimate."
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">
        📧 AI Email <span>Detector</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-subtitle">
    An AI-powered machine learning system that analyzes email
    content and classifies messages as <b>SPAM</b> or
    <b>LEGITIMATE</b>. The system uses a hybrid TF-IDF approach
    with Logistic Regression and is trained using thousands of
    real email samples.
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# EMAIL INPUT
# =========================================================

st.markdown(
    '<div class="section-title">✉️ Email Content</div>',
    unsafe_allow_html=True,
)

email_text = st.text_area(
    "Paste the email you want to analyze",
    value=st.session_state.email_text,
    height=230,
    label_visibility="collapsed",
    placeholder=(
        "Paste the complete email content here...\n\n"
        "Example:\n"
        "Congratulations! You have won a special prize..."
    ),
)

st.session_state.email_text = email_text


# =========================================================
# COUNTERS
# =========================================================

word_count = len(email_text.split())
character_count = len(email_text)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">Words</div>
    <div class="metric-value">{word_count}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">Characters</div>
    <div class="metric-value">{character_count}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
<div class="metric-card">
    <div class="metric-label">Test Accuracy</div>
    <div class="metric-value">98.87%</div>
</div>
""",
        unsafe_allow_html=True,
    )


st.write("")


# =========================================================
# ACTION BUTTONS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    analyze_clicked = st.button(
        "🔍 Analyze Email",
        use_container_width=True,
        type="primary",
    )

with col2:
    st.button(
        "🎉 Spam Example",
        use_container_width=True,
        on_click=set_spam_example,
    )

with col3:
    st.button(
        "📨 Legitimate Example",
        use_container_width=True,
        on_click=set_legitimate_example,
    )

with col4:
    st.button(
        "🗑️ Clear",
        use_container_width=True,
        on_click=clear_email,
    )


# =========================================================
# CLASSIFICATION
# =========================================================

if analyze_clicked:

    text = st.session_state.email_text.strip()

    if not text:

        st.warning("Please enter an email before analyzing.")

    else:

        with st.spinner("Analyzing email with AI..."):

            try:

                label, confidence = clf.predict(text)

                st.session_state.result = {
                    "label": label.lower(),
                    "confidence": confidence,
                }

            except Exception as exc:

                st.error(
                    f"Unable to analyze the email: {exc}"
                )

                st.session_state.result = None

# =========================================================
# RESULT
# =========================================================

if st.session_state.result:

    label = st.session_state.result["label"]
    confidence = st.session_state.result["confidence"]

    if label == "spam":

        st.error(
            f"""
🚨 LIKELY SPAM

AI Confidence: {confidence:.1%}

This email contains language and patterns that resemble spam
messages found in the training data. Be careful with links,
attachments, payment requests, and requests for personal information.
"""
        )

    else:

        st.success(
            f"""
✅ LIKELY LEGITIMATE

AI Confidence: {confidence:.1%}

This email resembles legitimate messages according to the trained
machine learning model. However, classification is based on email
content and does not guarantee that the sender or message is authentic.
"""
        )

# =========================================================
# MODEL DETAILS
# =========================================================

st.write("")

st.markdown(
    "### 🔬 About This Model"
)

st.markdown(
    """
**Hybrid TF-IDF** combines word-level and character-level
text features to capture both meaningful phrases and
smaller text patterns.

**Logistic Regression** uses these features to classify
an email as spam or legitimate.

The latest evaluation on the original held-out test set
achieved **98.87% accuracy**, with **99.77% spam precision**,
**96.63% spam recall**, and **98.17% spam F1 score**.
"""
)

# =========================================================
# DISCLAIMER
# =========================================================

st.write("")

st.info(
    "⚠️ This tool classifies email content based on patterns "
    "learned from the training data. It does not verify the "
    "real identity of the sender and does not perform SPF, "
    "DKIM, DMARC, or mail-server authentication checks."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
    AI Email Detector • Machine Learning Project<br>
    Hybrid TF-IDF + Logistic Regression
</div>
""",
    unsafe_allow_html=True,
)