import shap
import matplotlib.pyplot as plt
import pandas as pd

class SHAPExplainer:
    def __init__(self, model, X):
        self.model = model
        self.X = X
        self.explainer = None
        self.shap_values = None
        
    def calculate_shap(self):
        # TreeExplainer for Trees, LinearExplainer for Linear, Kernel for others
        model_type = type(self.model).__name__
        
        if "LGBM" in model_type or "RandomForest" in model_type or "XGB" in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        elif "Ridge" in model_type or "Lasso" in model_type:
            self.explainer = shap.LinearExplainer(self.model, self.X)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X, 50))
            
        self.shap_values = self.explainer.shap_values(self.X)
        
    def plot_summary(self):
        fig = plt.figure()
        shap.summary_plot(self.shap_values, self.X, show=False)
        return fig
        
    def plot_beeswarm(self):
        fig = plt.figure()
        shap.plots.beeswarm(self.explainer(self.X), show=False)
        return fig
