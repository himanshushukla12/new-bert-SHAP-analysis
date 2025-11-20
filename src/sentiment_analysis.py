from transformers import pipeline
import pandas as pd
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, truncation=True, max_length=512)

    def predict(self, texts: list[str]) -> list[int]:
        """
        Returns integer sentiment score 1-5.
        """
        # Process in batches to avoid memory issues if list is huge
        results = []
        # Simple loop for clarity, batching could be added
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            preds = self.pipe(batch)
            # preds example: [{'label': '5 stars', 'score': 0.9}]
            scores = [int(p['label'].split()[0]) for p in preds]
            results.extend(scores)
        return results
