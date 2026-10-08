# 📱 SMS Spam Detection Pro — NLP & ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.4%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Profile](https://img.shields.io/badge/Author-gouravdutta2004-181717?logo=github&logoColor=white)](https://github.com/gouravdutta2004)

An end-to-end, production-grade Natural Language Processing (NLP) and Machine Learning system that classifies SMS messages into **Spam** or **Ham (Legitimate)**. Built with a calibrated Linear SVM pipeline, group-aware cross-validation to prevent data leakage, an automated test suite, and an interactive Streamlit web application.

---

## 🌟 Key Highlights

* **Leak-Free ML Pipeline:** Feature extraction (`TfidfVectorizer`) and modeling (`LinearSVC`) are encapsulated in an end-to-end `sklearn.pipeline.Pipeline`.
* **Group-Aware Cross-Validation:** Uses 5-Fold `StratifiedGroupKFold` keyed on raw text to prevent duplicate/repeated SMS messages from crossing training and validation boundaries.
* **Calibrated Probabilities:** Wraps `LinearSVC` with `CalibratedClassifierCV` to yield reliable confidence scores and probabilities.
* **Interactive Streamlit UI:** Single message classifier with quick presets, bulk CSV batch processor with export capability, and live benchmark/EDA charts.
* **Tested & Reliable:** Complete `unittest` suite testing preprocessing normalization, dataset integrity, artifact loading, and probabilistic inference.

---

## 📐 Architecture & Workflow

```
Raw SMS Text
    ↓
Text Cleaning (src/preprocessing.py)
    • Lowercase conversion
    • Punctuation / symbol removal
    • Whitespace normalization
    ↓
TF-IDF Vectorization (TfidfVectorizer)
    • Unigrams + Bigrams (1, 2)
    • max_features = 5000
    ↓
Calibrated Linear Classifier (CalibratedClassifierCV + LinearSVC)
    • Stratified Grouped 5-Fold CV model selection
    • Calibrated spam probability output
    ↓
Streamlit Dashboard / Batch CSV / REST API
```

---

## 📊 Model Benchmarks & Evaluation

All candidate models were evaluated on the training partition using **5-Fold Stratified Grouped Cross-Validation** (zero duplicate leakage). The winning pipeline was evaluated on a group-disjoint hold-out test set:

| Model Architecture | CV Accuracy | CV Precision | CV Recall | CV F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** 🏆 | **98.41%** | **95.99%** | **91.95%** | **93.92%** |
| Random Forest | 97.31% | 99.19% | 80.57% | 88.90% |
| Multinomial Naive Bayes | 96.95% | 100.00% | 77.22% | 87.09% |
| Logistic Regression | 96.52% | 99.79% | 74.21% | 85.05% |

### 🎯 Final Hold-Out Test Set Results (Selected Pipeline)

* **Accuracy:** `99.10%`
* **Precision (Spam):** `97.30%`
* **Recall (Spam):** `96.00%`
* **F1-Score (Spam):** `96.64%`
* **ROC AUC:** `0.9945`

---

## 🗂️ Project Structure

```
detect-spam-nlp/
├── .streamlit/
│   └── config.toml               # Streamlit theme & server configuration
├── .vscode/
│   ├── launch.json               # One-click VS Code run & debug configs
│   └── settings.json             # Python interpreter & test explorer settings
├── data/
│   ├── SMSSpamCollection         # UCI SMS Spam Collection dataset (5,574 samples)
│   └── dataset_info.txt          # Dataset license and documentation
├── models/
│   ├── pipeline.joblib           # Pretrained scikit-learn pipeline artifact
│   └── metrics.json              # Evaluated CV and holdout metrics
├── notebooks/
│   └── spam_detection_nlp.ipynb  # Interactive walkthrough notebook
├── src/
│   ├── preprocessing.py          # Text normalization and cleaning logic
│   └── train.py                  # End-to-end model training & CV benchmarking
├── tests/
│   └── test_pipeline.py          # Automated test suite (unittest)
├── app.py                        # Streamlit interactive web application
├── requirements.txt              # Project dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/gouravdutta2004/SmsSpamDetection.git
cd SmsSpamDetection

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Application

```bash
# Using active venv
streamlit run app.py

# Or directly with python
python -m streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to interact with the dashboard.

### 3. Run Model Training Pipeline

```bash
python src/train.py
```

### 4. Run Automated Tests

```bash
python -m unittest tests/test_pipeline.py
```

---

## 💻 Running in Visual Studio Code

1. Open the project folder in VS Code (`File > Open Folder...`).
2. Press `Cmd + Shift + D` (or `Ctrl + Shift + D`) to open **Run & Debug**.
3. Select **`Run Streamlit Web App`** and press `F5`.
4. Open the in-editor preview using `Cmd + Shift + P` > **`Simple Browser: Show`** > enter `http://localhost:8501`.

---

## 📦 Dataset Information

* **Source:** [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
* **Total Messages:** `5,574`
* **Legitimate (Ham):** `4,827` (86.6%)
* **Spam:** `747` (13.4%)
* **Parsing Note:** Parsed with `quoting=csv.QUOTE_NONE` to ensure all 5,574 messages are loaded without corruption from unescaped quotes.

---

## 👨‍💻 Author

**Gourav Dutta**  
* GitHub: [@gouravdutta2004](https://github.com/gouravdutta2004)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
