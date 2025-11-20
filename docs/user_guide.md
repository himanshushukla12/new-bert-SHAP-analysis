# User Guide

## Getting Started

1.  **Prepare Data**: Ensure you have a CSV file with columns: `review_text`, `rating`. Optional: `review_id`, `app_name`.
2.  **Launch App**: Run `streamlit run app/streamlit_app.py`.
3.  **Select Dataset**: Choose "Mental Health" or "Quick Commerce" or upload your own.

## Interpreting Outputs

### Topic Modeling
- **Intertopic Distance Map**: Shows how similar topics are.
- **Topic Words**: The key terms defining each topic.

### Sentiment Analysis
- Shows the distribution of calculated sentiments vs actual ratings.

### Model Performance
- **RMSE (Root Mean Squared Error)**: Lower is better.
- **R2 Score**: Higher is better (closer to 1.0).

### SHAP Explainability
- **Beeswarm Plot**:
    - Each dot is a review.
    - **Color**: Feature value (Sentiment on that topic). Red = Positive Sentiment, Blue = Negative Sentiment.
    - **X-axis**: SHAP value (Impact on Rating). Positive SHAP = Increases Rating, Negative SHAP = Decreases Rating.
    - *Example*: If "Delivery Speed" has high red dots on the right, it means positive sentiment about delivery speed strongly increases user satisfaction.
