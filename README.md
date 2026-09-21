# 📧 AI Email Detector

An AI-powered machine learning application that analyzes email content and classifies messages as **Spam** or **Legitimate**.

The project uses a **Hybrid TF-IDF feature extraction approach** with **Logistic Regression** and provides an interactive web interface built with Streamlit.

---

## 🚀 Features

- 📧 Spam and legitimate email classification
- 🤖 Machine learning-based text classification
- 🔤 Hybrid word-level and character-level TF-IDF
- 🧠 Logistic Regression classifier
- 📊 98.87% test accuracy
- 🎯 99.77% spam precision
- 🔍 96.63% spam recall
- 📈 98.17% spam F1 score
- 🧹 Email text preprocessing
- 📝 Word and character counters
- 🎉 Built-in spam example
- 📨 Built-in legitimate email example
- 🗑️ Clear input functionality
- 🌙 Professional dark-themed Streamlit interface
- 💾 Trained model saved using Joblib

---

## 🧠 How It Works

The application follows this machine learning workflow:

```text
Email Input
     ↓
Text Cleaning
     ↓
Word-Level TF-IDF
     +
Character-Level TF-IDF
     ↓
Combined Feature Representation
     ↓
Logistic Regression
     ↓
Spam / Legitimate