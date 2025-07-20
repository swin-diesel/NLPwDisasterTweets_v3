# NLP with Disaster Tweets – Kaggle Classification Competition

This project explores the application of **Natural Language Processing (NLP)** to classify tweets as related to **real disasters** or not, as a part of the Kaggle competition [*Natural Language Processing with Disaster Tweets*](https://www.kaggle.com/competitions/nlp-getting-started). The competition is designed to introduce competitors to NLP tasks and workflows as applied to binary classification.

---

## Overview

As social media platforms like Twitter (now X) continue to serve as a real-time reporting tool, disaster response agencies are increasingly interested in **automated systems** that can distinguish between **actual emergency alerts** and unrelated content.

The competition provides **10,000 hand-labeled tweets** and tasks participants with building a model that can **predict whether a tweet refers to a real disaster**.

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

### v7: Baseline Model — TF-IDF + XGBoost  
- Standard preprocessing (lowercasing, punctuation removal, stopwords)  
- Bag-of-words features with TF-IDF weighting  
- XGBoost for classification  

### v8: Enhanced Cleaning + TF-IDF + XGBoost  
- Added custom cleaning for **UK/US spelling normalization**  
- Improved text cleaning: extra whitespace stripping, noise reduction  
- Rebuilt TF-IDF matrix on a more normalized vocabulary  

### v9: Word2Vec Embeddings + XGBoost  
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

Each version generated a submission file and was evaluated on **Kaggle’s public leaderboard**, which uses a **fixed test set** for F1 score:
```
| Version | Pipeline Description       | Kaggle F1 Score |
|---------|----------------------------|-----------------|
| v7      | TF-IDF Baseline            | 0.56114         |
| v8      | Enhanced TF-IDF Cleaning   | 0.57370         |
| v9      | Word2Vec + XGBoost         | 0.76953         |
```
**v9 significantly outperformed** the others, showing that Word2Vec’s semantic representation captured more signal than sparse TF-IDF features.

---

## Lessons Learned & Future Improvements

- **Internal F1 ≠ Kaggle Leaderboard F1**  
  Local validation splits gave **higher F1 scores** than the Kaggle leaderboard. This is a common pitfall when local validation sets do not mirror the distribution or difficulty of the competition's hidden test set. Future versions should use **stratified k-fold CV** or a holdout set that simulates test conditions more realistically.

- **Word2Vec Outperformed TF-IDF**  
  Even without a neural model like BERT, the Word2Vec embeddings improved classification by capturing deeper semantic structure. This reinforces the value of vector-based representations over sparse features.

- **Advanced Text Cleaning Had Minimal Impact**  
  While British/American normalization and extra cleaning steps were valuable academically, they produced **minimal gain** in actual classification performance.

- **No Hyperparameter Tuning Was Performed**  
  All models used basic XGBoost parameters. Future iterations could benefit from **grid search**, **Bayesian optimization**, or **automated tuning** to push performance further.

---

## Repository Structure

```
/NLPwDisasterTweets
├── NLPwDisasterTweets_v7.R          # TF-IDF Baseline
├── NLPwDisasterTweets_v8.R          # Enhanced Preprocessing
├── NLPwDisasterTweets_v9.R          # Word2Vec Approach
├── uk-us-spelling-list.csv          # Dictionary for normalization
├── submission4.csv                  # From v7 – Score: 0.56114
├── submission5.csv                  # From v8 – Score: 0.57370
├── submission6.csv                  # From v9 – Score: 0.76953
└── README.md                        # This file
```

---

## How to Run This Project

1. Open the project in **RStudio**.
2. Run `renv::restore()` to install the required libraries.
3. Use `source("NLPwDisasterTweets_vX.R")` to run any version (replace `X` with 7, 8, or 9).
4. Generated predictions will be saved as a submission CSV (e.g., `submission6.csv`).

---

## Author

**Michael Swindle**
[Portfolio Website](https://michaelswindle.dev) | [GitHub Profile](https://github.com/swin-diesel) | [LinkedIn](https://linkedin.com/in/michael-swindle/)

---

## License

This project is licensed under the **MIT License**.

---

## Next Steps

- Compare results with **transformer-based models** (e.g., BERT via `reticulate`)
- Expand with **more advanced spell correction and typo normalization**
- Add **model ensemble** between TF-IDF and Word2Vec pipelines
