import pandas as pd
import numpy as np
from typing import Tuple

def create_features(df: pd.DataFrame, topics: list, sentiment_scores: list, topic_probs=None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs X (features) and y (target).
    
    Strategy:
    We want to predict 'rating' (y) using 'sentiment towards topics' (X).
    
    Input:
    - df: DataFrame with 'rating'
    - topics: List of topic IDs for each document (from BERTopic)
    - sentiment_scores: List of sentiment scores (1-5) for each document
    - topic_probs: (Optional) Probability matrix of shape (n_docs, n_topics)
    
    Approach:
    If we have topic probabilities, we can create a dense feature matrix.
    Feature_j = Probability(Topic_j) * Sentiment_Score
    
    This assumes the sentiment of the *entire* review applies to *all* topics mentioned, 
    weighted by how much the topic is present.
    
    If topic_probs is None or we just use the dominant topic:
    We can One-Hot Encode the dominant topic and multiply by sentiment.
    Or create columns for each topic, set value = sentiment if dominant, else 0 (or neutral).
    
    Let's use the Probability * Sentiment approach as it's more continuous and robust.
    """
    
    y = df['rating']
    
    # If topic_probs is available (n_samples, n_topics)
    if topic_probs is not None:
        # We need to map the columns of topic_probs to actual topic IDs.
        # BERTopic usually returns probabilities for topics in order of index, but topic -1 (outlier) might be handled differently.
        # Let's assume topic_probs corresponds to topics 0, 1, 2... (excluding -1 usually, or including it).
        # Actually BERTopic.transform returns probs for all topics.
        
        # Let's simplify: X = topic_probs * sentiment_score (broadcasted)
        # But sentiment_score is 1-5. 
        # If a user is angry (1 star) about 'Delivery' (prob 0.9), the feature 'Delivery' gets 0.9 * 1 = 0.9.
        # If a user is happy (5 star) about 'Delivery' (prob 0.9), the feature 'Delivery' gets 0.9 * 5 = 4.5.
        # This seems linear and reasonable for regression.
        
        # Normalize sentiment to be centered? Maybe not needed for tree models.
        
        sentiments = np.array(sentiment_scores).reshape(-1, 1)
        X_values = topic_probs * sentiments
        
        # Create DataFrame
        # We need column names. Let's assume topic indices 0..N
        # Note: topic_probs might not include topic -1.
        num_topics = topic_probs.shape[1]
        col_names = [f"Topic_{i}" for i in range(num_topics)]
        X = pd.DataFrame(X_values, columns=col_names)
        
    else:
        # Fallback: Dominant topic only
        # Create a matrix of zeros
        unique_topics = sorted(list(set(topics)))
        if -1 in unique_topics: unique_topics.remove(-1)
        
        X = pd.DataFrame(0, index=range(len(df)), columns=[f"Topic_{t}" for t in unique_topics])
        
        for i, (t, s) in enumerate(zip(topics, sentiment_scores)):
            if t != -1:
                X.loc[i, f"Topic_{t}"] = s
                
    return X, y
