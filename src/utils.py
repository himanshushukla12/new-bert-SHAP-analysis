import pandas as pd
import numpy as np

def generate_dummy_data(domain="mha", n_samples=1000):
    """
    Generates dummy data for testing.
    """
    if domain == "mha":
        texts = [
            "This app helps me sleep better.",
            "The meditation sessions are too short.",
            "Great for anxiety tracking.",
            "Subscription is too expensive.",
            "I love the soothing sounds.",
            "The UI is confusing and buggy.",
            "Helped me stop smoking.",
            "Customer support never replies.",
            "Best mental health app ever.",
            "Not worth the money."
        ] * (n_samples // 10)
    else:
        texts = [
            "Delivery was super fast!",
            "Items were missing from my order.",
            "Rider was rude.",
            "Vegetables were not fresh.",
            "Great discounts on groceries.",
            "App crashes when paying.",
            "Refund process is a nightmare.",
            "Love the 10 minute delivery.",
            "Packaging was damaged.",
            "Too many delivery fees."
        ] * (n_samples // 10)
        
    ratings = np.random.randint(1, 6, size=len(texts))
    
    df = pd.DataFrame({
        "review_text": texts,
        "rating": ratings,
        "review_id": range(len(texts)),
        "app_name": "TestApp"
    })
    
    return df
