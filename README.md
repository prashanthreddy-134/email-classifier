# 📧 Email Classifier

A machine learning based **Email Spam Classifier** that automatically analyzes email text and classifies messages as **Spam** or **Ham (Not Spam)**.

The project includes a trained machine learning model and an interactive **Streamlit web application** for real-time email classification.

---

# 📌 Project Overview

Email spam is a common problem where unwanted or malicious messages are sent to users.

This project uses **Natural Language Processing (NLP)** and machine learning to analyze email content and classify it into:

- 🚨 Spam
- ✅ Ham / Not Spam

### Classification Pipeline

```text
Email Text
    ↓
Text Preprocessing
    ↓
Feature Extraction
    ↓
Machine Learning Model
    ↓
Spam / Ham Prediction
    ↓
Confidence / Result
```

---

# 🚀 Features

- 📧 Email text classification
- 🚨 Spam detection
- ✅ Ham / Not Spam detection
- 🧠 Machine learning based prediction
- 📝 Natural Language Processing
- 🔤 Text feature extraction
- 🌐 Streamlit web application
- ⚡ Real-time classification
- 📊 Prediction results
- 💾 Pre-trained model included
- 📦 Easy local installation
- 🛠️ Training script included

---

# 🧠 Machine Learning Approach

The project uses a supervised machine learning approach for email classification.

The training pipeline processes email text and converts it into numerical features that can be understood by the machine learning classifier.

### Model Pipeline

```text
Raw Email
    ↓
Text Cleaning
    ↓
Text Vectorization
    ↓
Feature Representation
    ↓
Machine Learning Classifier
    ↓
Prediction
    ↓
Spam / Ham
```

The trained model is stored in:

```text
models/email_classifier.joblib
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Joblib | Model serialization |
| Streamlit | Web application |
| NLP | Email text processing |
| CSV | Dataset storage |

---

# 📂 Project Structure

```text
email-classifier/
│
├── data/
│   └── emails.csv
│
├── models/
│   └── email_classifier.joblib
│
├── app.py
├── classifier.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 💻 How to Run the Project

Follow these steps after downloading or cloning the repository.

---

## 1️⃣ Install Python

Install **Python 3.9 or newer**.

Check your Python installation:

```bash
python --version
```

If `python` does not work on Windows, try:

```bash
py --version
```

---

# 2️⃣ Clone the GitHub Repository

Open **PowerShell**, **Command Prompt**, or a terminal.

Run:

```bash
git clone https://github.com/prashanthreddy-134/email-classifier.git
```

---

# 3️⃣ Enter the Project Folder

```bash
cd email-classifier
```

---

# 4️⃣ Create a Virtual Environment

Creating a virtual environment is recommended.

```bash
python -m venv venv
```

---

# 5️⃣ Activate the Virtual Environment

## Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

## Windows Command Prompt

```cmd
venv\Scripts\activate
```

After activation, you should see:

```text
(venv)
```

before your terminal prompt.

---

# 6️⃣ Install Dependencies

Install all required packages:

```bash
python -m pip install -r requirements.txt
```

Wait until the installation finishes successfully.

---

# ▶️ Run the Email Classifier

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

You should see something similar to:

```text
Local URL: http://localhost:8501
```

Open the URL in your browser:

```text
http://localhost:8501
```

---

# ⚡ Quick Start

If Python is already installed and you want to run the project quickly:

```bash
git clone https://github.com/prashanthreddy-134/email-classifier.git
cd email-classifier
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🪟 Windows PowerShell — Complete Setup

Run these commands **one at a time**:

```powershell
git clone https://github.com/prashanthreddy-134/email-classifier.git
```

```powershell
cd email-classifier
```

```powershell
python -m venv venv
```

```powershell
.\venv\Scripts\Activate.ps1
```

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🖥️ How to Use the Application

After the Streamlit application opens:

1. Enter or paste an email message.
2. Submit the email for analysis.
3. The application preprocesses the text.
4. The trained machine learning model analyzes the email.
5. The application displays the classification result.
6. The result identifies the email as **Spam** or **Ham**.

---

# 🧪 Train the Model

The project includes the training script:

```text
train.py
```

To train the model again:

```bash
python train.py
```

The trained model is saved to:

```text
models/email_classifier.joblib
```

> Training may take longer than running the already-trained application because the training process rebuilds the model from the dataset.

---

# 🔄 Retrain the Classifier

If you modify the dataset or training configuration, run:

```bash
python train.py
```

After training finishes, start the application again:

```bash
python -m streamlit run app.py
```

---

# 🧠 Use the Classifier Directly

The classification logic is implemented in:

```text
classifier.py
```

The Streamlit application uses this module to load the trained model and perform predictions.

---

# 🛑 Stop the Application

To stop the Streamlit application:

```text
Ctrl + C
```

---

# 📊 Dataset

The processed dataset is stored at:

```text
data/emails.csv
```

The dataset contains email examples used for training and evaluating the classifier.

The repository intentionally excludes large raw training archives and raw SpamAssassin corpus files that are not required to run the finished application.

---

# 🔐 Model

The trained model is included in:

```text
models/email_classifier.joblib
```

This allows users to run the Streamlit application without retraining the model first.

---

# 📦 Requirements

All required Python dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
python -m pip install -r requirements.txt
```

---

# ❗ Troubleshooting

## Python is not recognized

Try:

```bash
py --version
```

Then:

```bash
py -m pip install -r requirements.txt
```

And:

```bash
py -m streamlit run app.py
```

---

## Streamlit is not recognized

Instead of:

```bash
streamlit run app.py
```

use:

```bash
python -m streamlit run app.py
```

---

## Port 8501 is already in use

Run Streamlit on another port:

```bash
python -m streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

---

## Dependency Installation Problem

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Then reinstall the dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Model File Not Found

Make sure this file exists:

```text
models/email_classifier.joblib
```

If the model is missing, train it using:

```bash
python train.py
```

Then run:

```bash
python -m streamlit run app.py
```

---

# 📈 Project Workflow

```text
Email Dataset
      ↓
Data Cleaning
      ↓
Text Processing
      ↓
Feature Extraction
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Save Trained Model
      ↓
Streamlit Application
      ↓
User Email
      ↓
Spam / Ham Prediction
```

---

# 💡 Concepts Demonstrated

This project demonstrates practical knowledge of:

- Python
- Machine Learning
- Natural Language Processing
- Text preprocessing
- Feature extraction
- Supervised learning
- Classification
- Model training
- Model evaluation
- Model serialization
- Streamlit
- End-to-end ML application development

---

# 🔮 Future Improvements

Possible improvements include:

- 📧 Gmail integration
- 📬 Outlook integration
- 🤖 Deep learning based text classification
- 🧠 Transformer-based classification
- 📊 Advanced model analytics
- 📈 Confusion matrix dashboard
- 📉 Precision/Recall visualization
- 🔍 Explainable AI
- 🌐 Cloud deployment
- 📱 Mobile-friendly interface
- 🔐 Phishing URL detection
- 🛡️ Malicious attachment detection
- 🧩 Multi-language spam detection

---

# ⚠️ Limitations

The classifier's performance depends on the dataset used for training.

Email patterns can change over time, and new types of spam may not be represented in the training data.

Therefore, the model should be treated as a machine learning classification system rather than a guaranteed spam-detection mechanism.

---

# 👨‍💻 Author

## Prashanth Reddy S

**BE – Information Science & Engineering**

### Interests

- Machine Learning
- Artificial Intelligence
- Python
- Data Science
- Natural Language Processing
- Full-Stack Development
- Computer Vision

---

# 📜 License

This project is intended for educational and portfolio purposes.

---

# ⭐ GitHub Repository

https://github.com/prashanthreddy-134/email-classifier

If you find this project useful, consider giving the repository a ⭐.