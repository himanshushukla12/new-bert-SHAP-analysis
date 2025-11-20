# Architecture Overview

## System Flow

1.  **Data Ingestion**: Raw CSV files are loaded. The system auto-detects the domain (MHA or Quick Commerce).
2.  **Preprocessing**: Text is cleaned, tokenized, and POS-tagged.
3.  **Topic Modeling (BERTopic)**:
    *   Embeddings: `all-MiniLM-L6-v2`
    *   Clustering: HDBSCAN
    *   Dim Reduction: UMAP
    *   Topics are generated and labeled.
4.  **Sentiment Analysis**:
    *   Model: `nlptown/bert-base-multilingual-uncased-sentiment`
    *   Each review gets a sentiment score (1-5).
5.  **Feature Engineering**:
    *   Reviews are mapped to their dominant topics.
    *   For each document, we associate the sentiment score with the identified topics.
    *   (Alternative approach per paper: Topic-level sentiment might be aggregated, but usually, we use the document's sentiment as a feature for the topics present, or we calculate specific sentiment per topic if the model allows. Here we will follow a standard approach: The features `X` are the sentiment scores weighted by topic probability or simply the sentiment score assigned to the dominant topic columns).
    *   *Refinement*: The prompt says "Build ML models using topic-level sentiment as independent variables". This implies for a given user (or review), we need variables representing their sentiment towards specific topics. Since a single review might only touch on 1-2 topics, the feature matrix might be sparse or we might need to aggregate at a user level if multiple reviews exist. However, usually, these datasets are single-review per user.
    *   *Strategy*: `X` = Matrix where columns are Topics. Values are the sentiment score of the review *IF* the review belongs to that topic (or weighted by probability). If a review does not belong to a topic, the value is 0 or neutral (3).
6.  **Machine Learning**:
    *   Target `y`: User Rating (1-5).
    *   Models: Ridge, Lasso, NB, RF, KNN, LightGBM.
    *   Selection: Best RMSE/MAE/R2.
7.  **Explainability**:
    *   SHAP is applied to the best model to find which Topics (features) drive the Rating (target) the most.

## Module Responsibilities

- `src/data_loading.py`: Load CSV, detect domain.
- `src/preprocessing.py`: NLP cleaning.
- `src/topic_modeling.py`: BERTopic wrapper.
- `src/sentiment_analysis.py`: HuggingFace pipeline wrapper.
- `src/features.py`: Construct X and y.
- `src/models.py`: Train and evaluate regressors/classifiers.
- `src/explainability.py`: SHAP analysis.
