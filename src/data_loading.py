import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

def detect_domain(df: pd.DataFrame) -> str:
    """
    Auto-detects domain based on keywords in text.
    """
    # Simple heuristic: check for keywords in a sample of text
    text_sample = " ".join(df['review_text'].astype(str).head(100).tolist()).lower()
    
    mha_keywords = ['meditation', 'anxiety', 'therapy', 'mood', 'sleep', 'calm', 'mental']
    qc_keywords = ['delivery', 'rider', 'grocery', 'order', 'refund', 'late', 'item', 'fresh']
    
    mha_score = sum(1 for k in mha_keywords if k in text_sample)
    qc_score = sum(1 for k in qc_keywords if k in text_sample)
    
    if mha_score > qc_score:
        return "mha"
    elif qc_score > mha_score:
        return "quick_commerce"
    else:
        return "unknown"

def load_data(file_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Loads CSV and detects domain.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")
    
    # Normalize columns to lowercase for checking
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Smart column detection
    text_candidates = ['review_text', 'review', 'text', 'body', 'comment', 'content', 'feedback']
    rating_candidates = ['rating', 'score', 'star', 'stars', 'label', 'grade']
    
    text_col = next((c for c in text_candidates if c in df.columns), None)
    rating_col = next((c for c in rating_candidates if c in df.columns), None)
    
    # If exact match not found, try partial match
    if not text_col:
        text_col = next((c for c in df.columns if any(x in c for x in ['review', 'text', 'body'])), None)
    if not rating_col:
        rating_col = next((c for c in df.columns if any(x in c for x in ['rating', 'score', 'star'])), None)
        
    if not text_col or not rating_col:
        raise ValueError(f"Could not automatically detect 'review' and 'rating' columns. Found: {list(df.columns)}. Please rename columns to 'review_text' and 'rating'.")
        
    # Rename to standard names
    df = df.rename(columns={text_col: 'review_text', rating_col: 'rating'})
    
    # Ensure numeric rating
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['review_text', 'rating'])
    
    domain = detect_domain(df)
    
    return df, domain
