import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loading import load_data, detect_domain
from src.preprocessing import preprocess_pipeline
from src.topic_modeling import TopicModeler
from src.sentiment_analysis import SentimentAnalyzer
from src.features import create_features
from src.models import ModelTrainer
from src.explainability import SHAPExplainer
from src.utils import generate_dummy_data

st.set_page_config(page_title="Dual Domain ML Framework", layout="wide")

st.title("Dual Domain ML Framework: Satisfaction Determinants")

# Sidebar
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Data Source", ["Upload CSV", "Use Dummy MHA Data", "Use Dummy Quick Commerce Data"])

df = None
domain = "unknown"

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Review CSV", type=["csv"])
    if uploaded_file:
        df, domain = load_data(uploaded_file)
elif data_source == "Use Dummy MHA Data":
    df = generate_dummy_data("mha")
    domain = "mha"
else:
    df = generate_dummy_data("quick_commerce")
    domain = "quick_commerce"

if df is not None:
    st.sidebar.success(f"Loaded {len(df)} reviews. Domain: {domain}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Topic Modeling", "Sentiment Analysis", "Model Performance", "Explainability"])
    
    with tab1:
        st.header("Dataset Overview")
        st.dataframe(df.head())
        st.write(f"**Total Reviews:** {len(df)}")
        st.write(f"**Average Rating:** {df['rating'].mean():.2f}")
        st.bar_chart(df['rating'].value_counts())

    # Pipeline Execution (Cached if possible, but for now direct)
    if st.sidebar.button("Run Pipeline"):
        with st.spinner("Preprocessing..."):
            df = preprocess_pipeline(df, domain)
            
        with st.spinner("Topic Modeling (BERTopic)..."):
            # Reduce samples for speed in demo if needed, but full run here
            topic_modeler = TopicModeler()
            topics, probs = topic_modeler.fit_transform(df['clean_text'].tolist())
            topic_info = topic_modeler.get_topic_info()
            
        with tab2:
            st.header("Topic Modeling Results")
            st.dataframe(topic_info)
            fig = topic_modeler.visualize_topics()
            if fig:
                st.plotly_chart(fig)
            else:
                st.warning("Not enough topics found to visualize. Try increasing dataset size or adjusting model parameters.")
            
        with st.spinner("Sentiment Analysis..."):
            sentiment_analyzer = SentimentAnalyzer()
            sentiment_scores = sentiment_analyzer.predict(df['clean_text'].tolist())
            df['sentiment_score'] = sentiment_scores
            
        with tab3:
            st.header("Sentiment Analysis")
            st.write("Sentiment Scores Distribution")
            st.bar_chart(pd.Series(sentiment_scores).value_counts())
            st.write("Sample Predictions")
            st.dataframe(df[['review_text', 'rating', 'sentiment_score']].head(10))
            
        with st.spinner("Feature Engineering & Modeling..."):
            # Handle probs being 1D or 2D
            if probs.ndim == 1:
                # If BERTopic returns 1D array (probabilities=False or only 1 topic?), 
                # but we set calculate_probabilities=True.
                # Sometimes it returns just the probability of the assigned topic.
                # We need the full probability matrix for the feature engineering strategy we chose.
                # If probs is 1D, we might need to rely on 'topics' (dominant topic) only.
                # For safety in this demo, let's check.
                X, y = create_features(df, topics, sentiment_scores, topic_probs=None) # Fallback to dominant
            else:
                X, y = create_features(df, topics, sentiment_scores, topic_probs=probs)
                
            trainer = ModelTrainer()
            results, best_model_name = trainer.train_and_evaluate(X, y)
            
        with tab4:
            st.header("Model Performance")
            st.write(f"**Best Model:** {best_model_name}")
            st.json(results)
            
        with st.spinner("Explainability (SHAP)..."):
            explainer = SHAPExplainer(trainer.best_model, X)
            explainer.calculate_shap()
            
        with tab5:
            st.header("SHAP Explainability")
            st.write("Global Feature Importance")
            st.pyplot(explainer.plot_summary())
            # st.write("Beeswarm Plot")
            # st.pyplot(explainer.plot_beeswarm()) # Beeswarm might fail if shap version issues, summary is safer
