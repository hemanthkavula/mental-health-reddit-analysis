# 🧠 Mental Health Post Analysis Using Machine Learning on Reddit Data

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20Logistic%20Regression-green)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-BERT%20%7C%20Word2Vec%20%7C%20TF--IDF-orange)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A comprehensive machine learning project analyzing **8,823 Reddit posts** from mental health communities to identify patterns, classify crisis-level content, and extract meaningful insights using NLP and unsupervised learning techniques.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Results](#results)
- [Technical Stack](#technical-stack)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Model Performance](#model-performance)

---

## 🎯 Project Overview

Mental health discussions on social media platforms provide valuable insights into community well-being. This project leverages **Natural Language Processing (NLP)** and **Machine Learning** to analyze Reddit posts from mental health communities, aiming to:

- Classify posts indicating severe mental health concerns (depression, suicidal ideation)
- Identify common themes using topic modeling
- Analyze sentiment trends and engagement metrics
- Provide actionable insights for public health monitoring

---

## ✨ Key Features

### 🔍 Data Collection & Ethics
- Collected **8,823 posts** from 6 mental health subreddits using Reddit API
- Implemented ethical data handling with anonymization
- Full compliance with Reddit's API terms of service

### 🧹 Advanced Text Preprocessing
- NLTK-based text cleaning pipeline (tokenization, lemmatization, stopword removal)
- Multiple embedding techniques: **TF-IDF**, **Word2Vec**, **BERT**
- Handled class imbalance and missing data

### 🤖 Machine Learning Models
- **Supervised Learning:** Logistic Regression (ROC-AUC: 0.77), **Random Forest (Selected)** (72% accuracy)
- **Unsupervised Learning:** K-Means Clustering, **LDA Topic Modeling (Selected)**

### 📊 Comprehensive EDA
- Temporal analysis of posting patterns
- Sentiment distribution using TextBlob
- Engagement metrics correlation
- Outlier detection (48 identified)

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| Source | Reddit API (6 mental health subreddits) |
| Size | 8,823 posts |
| Keywords | depression, anxiety, therapy, mental health, stress, suicidal |
| Time Period | 2025 data |

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Recall | ROC-AUC |
|-------|----------|--------|----------|
| Logistic Regression | 73% | 0.38 | **0.77** |
| **Random Forest** ✅ | 72% | **0.44** | 0.70 |

**Selected:** Random Forest - Higher recall minimizes false negatives in critical mental health detection.

### Key Findings
- Most discussed keyword: "therapy"
- Sentiment: Slightly negative to neutral
- Engagement: Emotionally charged posts receive higher upvotes
- Topics identified: Crisis narratives vs. Coping strategies

---

## 🛠️ Technical Stack

| Category | Technologies |
|----------|-------------|
| Programming | Python 3.8+ |
| Data Collection | PRAW (Reddit API) |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, Gensim, HuggingFace (BERT), TextBlob |
| ML | Scikit-learn |
| Visualization | Matplotlib, Seaborn |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone repository
git clone https://github.com/hemanthkavula/mental-health-reddit-analysis.git
cd mental-health-reddit-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 💻 Usage

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/mentalhealth.ipynb
```

### Key Code Examples

**Text Preprocessing:**
```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return ' '.join([WordNetLemmatizer().lemmatize(w) for w in tokens])
```

**Model Training:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['clean_text'])
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

---

## 📊 Model Performance Details

### Confusion Matrix (Random Forest)
```
                Predicted
              General  Critical
Actual
General        1009     152
Critical        340     262
```

- **True Positives:** 262 critical posts correctly identified
- **False Negatives:** 340 (minimized for mental health safety)
- **True Negatives:** 1009
- **False Positives:** 152

---

## 🔮 Future Enhancements

- [ ] Real-time monitoring dashboard (Streamlit)
- [ ] Deep learning models (LSTM, Transformers)
- [ ] Multi-class classification (anxiety, depression, general)
- [ ] REST API deployment
- [ ] Model explainability (LIME/SHAP)
- [ ] Automated alert system

---

## 📁 Project Structure

```
mental-health-reddit-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── mentalhealth.ipynb
├── docs/
│   ├── Final_project.pdf
│   └── finalproject.pptx
└── data/
    └── .gitkeep
```

---

## 📄 License

MIT License - see LICENSE file for details

---

## 👤 Contact

**Hemanth Kavula**
- 📧 Email: hemanthkavula2001@gmail.com
- 💼 LinkedIn: linkedin.com/in/hemanthkavula
- 🌐 GitHub: github.com/hemanthkavula
- 📍 Location: Pitman, NJ

---

<div align="center">

### ⭐ If you find this project helpful, please give it a star!

**Built with ❤️ for mental health awareness and data science**

</div>