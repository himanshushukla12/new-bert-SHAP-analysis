import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loading import detect_domain
from src.preprocessing import clean_text
from src.features import create_features

def test_preprocessing():
    text = "Hello World! This is a test."
    cleaned = clean_text(text)
    assert cleaned == "hello world this is a test"

def test_domain_detection():
    df_mha = pd.DataFrame({'review_text': ['meditation anxiety sleep'], 'rating': [5]})
    assert detect_domain(df_mha) == "mha"
    
    df_qc = pd.DataFrame({'review_text': ['delivery rider grocery'], 'rating': [5]})
    assert detect_domain(df_qc) == "quick_commerce"

def test_feature_creation():
    df = pd.DataFrame({'rating': [1, 5]})
    topics = [0, 1]
    sentiments = [1, 5]
    # Test without probs
    X, y = create_features(df, topics, sentiments, topic_probs=None)
    assert X.shape == (2, 2) # Topic_0, Topic_1
    assert y.shape == (2,)
