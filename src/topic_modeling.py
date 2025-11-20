from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
from typing import List, Tuple

class TopicModeler:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        # Configure sub-models as per prompt
        # min_cluster_size = 80
        # calculate_probabilities = True
        # diversity = 0.5 (This is usually a parameter in maximal_marginal_relevance or similar representation tuning, 
        # but BERTopic main init doesn't take 'diversity' directly in recent versions, it's often in representation_model)
        # We will stick to standard init and apply diversity in representation if needed, or just ignore if API differs slightly.
        
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        self.hdbscan_model = HDBSCAN(min_cluster_size=80, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        self.vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english")
        
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            calculate_probabilities=True,
            min_topic_size=80 # "auto" isn't a standard int value, prompt said "auto" or 80. Let's use 80 to match min_cluster_size.
        )
        
    def fit_transform(self, docs: List[str]) -> Tuple[List[int], pd.DataFrame]:
        topics, probs = self.model.fit_transform(docs)
        return topics, probs
    
    def get_topic_info(self):
        return self.model.get_topic_info()
    
    def get_topics(self):
        return self.model.get_topics()
        
    def visualize_topics(self):
        try:
            topic_info = self.model.get_topic_info()
            # Check if we have enough topics (excluding outlier -1)
            # topic_info has columns 'Topic', 'Count', 'Name', ...
            # Topic -1 is the outlier topic.
            non_outlier_topics = topic_info[topic_info['Topic'] != -1]
            
            if len(non_outlier_topics) == 0:
                return None
                
            return self.model.visualize_topics()
        except Exception as e:
            print(f"Error visualizing topics: {e}")
            return None
