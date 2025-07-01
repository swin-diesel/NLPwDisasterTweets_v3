# NLP with Disaster Tweets – Kaggle Classification Challenge

This project explores the application of **Natural Language Processing (NLP)** to classify tweets as related to **real disasters** or not, using a dataset provided by Kaggle. The challenge serves as an excellent introduction to NLP tasks in a binary classification context.

---

## Overview

As Twitter continues to serve as a real-time reporting tool, disaster response agencies are increasingly interested in **automated systems** that can distinguish between **actual emergency alerts** and unrelated content.  

Kaggle’s “NLP with Disaster Tweets” challenge provides **10,000 hand-labeled tweets** and tasks participants with building a model that can **predict whether a tweet refers to a real disaster**.  

This project:
- **Tests multiple text preprocessing pipelines**
- **Applies different feature extraction methods** (TF-IDF, Word2Vec)
- **Compares performance using the XGBoost classifier**
- Measures output using **F1 score**, the competition’s evaluation metric.

---

## Problem Statement

While humans can intuitively spot sarcasm, metaphor, or exaggeration, machines require **explicit training**. For example, the tweet:

> "Just got my morning coffee and my house is ablaze ☕🔥"

...uses "ablaze" metaphorically. A model needs to learn that this is **not a real disaster**, despite the use of alarming language.

The goal is to:
- **Develop a model** that can distinguish real disaster-related tweets from figurative or unrelated content.
- Do so using **realistic preprocessing and feature engineering pipelines** suitable for production-oriented NLP.

---

## Approach & Methodology

Three experimental versions were created to test different techniques:

### ✅ v7: Baseline Model — TF-IDF + XGBoost  
- Standard preprocessing (lowercasing, punctuation removal, stopwords)  
- Bag-of-words features with TF-IDF weighting  
- XGBoost for classification  

### ✅ v8: Enhanced Cleaning + TF-IDF + XGBoost  
- Added custom cleaning for **UK/US spelling normalization**  
- Improved text cleaning: extra whitespace stripping, noise reduction  
- Rebuilt TF-IDF matrix on a more normalized vocabulary  

### ✅ v9: Word2Vec Embeddings + XGBoost  
- Replaced TF-IDF with **Word2Vec vector averaging**  
- Captures **semantic similarity** instead of just term frequency  
- Same classifier (XGBoost), new feature set  

---

## Key Competencies & Skills Demonstrated

- **Natural Language Preprocessing** – stopword removal, lemmatization, tokenization  
- **Feature Engineering** – TF-IDF vectorization and Word2Vec embeddings  
- **Supervised Learning (XGBoost)** – tuned parameters for classification  
- **Evaluation using F1 Score** – balanced metric to assess both precision and recall  
- **Pipeline Variability Testing** – repeated runs and comparison of output drift  
- **Reproducibility Tools** – Uses `renv` for R dependency tracking  

---

## Results

✅ **v7 (TF-IDF)** produced stable and repeatable results across multiple runs  
⚠️ **v8 (Enhanced Cleaning)** introduced slight **variance** in submission output, likely due to **vocabulary shift in TF-IDF**  
🎲 **v9 (Word2Vec)** showed expected randomness due to underlying stochastic nature of the embeddings  

Each script prints **F1 scores**, **confusion matrices**, and generates a **submission file** (`submissionN.csv`).

---

## Lessons Learned & Future Improvements

- **TF-IDF vocab instability** can lead to output drift even with fixed seed — consider vocab locking or using `TermDocumentMatrix()` with explicit dictionary  
- **Word2Vec introduces run-to-run variability** — multiple runs needed to average performance  
- Custom cleaning (e.g., **British/American spelling unification**) is useful but should be paired with fixed vocab tools  
- Consider adding **grid search or automated tuning** for future improvements  

---

## Repository Structure

```
/NLPwDisasterTweets
├── NLPwDisasterTweets_v7.R          # TF-IDF Baseline
├── NLPwDisasterTweets_v8.R          # Enhanced Preprocessing
├── NLPwDisasterTweets_v9.R          # Word2Vec Approach
├── uk-us-spelling-list.csv          # Dictionary for normalization
├── submission4.csv                  # From v7 - Stable
├── submission5.csv                  # From v8 - Drift Detected
├── submission6.csv                  # From v9 - Semantic Vectors
└── README.md                        # This file
```

---

## How to Run This Project

1. Open the project in **RStudio**.
2. Run `renv::restore()` to install the required libraries.
3. Use `source("NLPwDisasterTweets_vX.R")` to run any version (replace `X` with 7, 8, or 9).
4. Generated predictions will be saved as a submission CSV (e.g., `submission5.csv`).

---

## Author

**Michael Swindle**  
[GitHub Profile (Placeholder)](#) | [LinkedIn (Placeholder)](#)

---

## License

This project is licensed under the **MIT License**.

---

## Next Steps

📌 Compare results with **transformer-based models** (e.g., BERT via `reticulate`)
📌 Expand with **more advanced spell correction and typo normalization**
📌 Add **model ensemble** between TF-IDF and Word2Vec pipelines
📌 Package code for reproducible submission automation
