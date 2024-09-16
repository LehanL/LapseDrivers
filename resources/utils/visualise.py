import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


class ModelPerformancePlotter:
    def __init__(self, y_pred, y_prob, y_actual, metrics_dict):
        """
        Initialises the ModelPerformancePlotter class.

        Parameters:
        - y_pred (array-like): Predicted labels.
        - y_actual (array-like): Actual labels.
        - y_prob (array-like): Predicted probabilities.
        - metrics_dict (dict): Dictionary containing performance metrics.
        """
        self.y_pred = y_pred
        self.y_actual = y_actual
        self.y_prob = y_prob
        self.metrics_dict = metrics_dict


    def plot_performance(self, storage_path=None):
        """
        Plots performance metrics and saves the figure to the specified path.

        Parameters:
        - save_path (str, optional): Path to save the plot figure.
        """
        # Extract metrics and scores from the metrics dictionary
        metrics = list(self.metrics_dict.keys())
        scores = list(self.metrics_dict.values())
        
        # place extracted metrics in dataframe
        df_performance = pd.DataFrame({
            'Metric': metrics,
            'Score': scores
        })

        # Initialise figure
        plt.figure(figsize=(18, 6))
        
        # Subplot 1: Performance Metrics Bar Plot
        plt.subplot(1, 3, 1)
        sns.barplot(x='Metric', y='Score', data=df_performance)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        plt.xlabel('Metric')
        plt.ylabel('Score')

        # Subplot 2: ROC Curve
        fpr, tpr, _ = roc_curve(self.y_actual, self.y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.subplot(1, 3, 2)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        # Subplot 3: CAP Curve
        sorted_indices = np.argsort(self.y_prob)[::-1]
        sorted_y = np.array(self.y_actual)[sorted_indices]
        total = len(sorted_y)
        cum_pos_rate = np.cumsum(sorted_y) / sum(sorted_y)  # Cumulative positive rate
        x = np.arange(1, total + 1) / total  # Percentage of total population
        
        plt.subplot(1, 3, 3)
        plt.plot(x, cum_pos_rate, color='blue', lw=2, label='CAP curve')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        
        # Draw the "perfect model" line
        plt.plot([0, sum(sorted_y) / total, 1], [0, 1, 1], color='green', linestyle='--', label='Perfect Model')

        plt.xlabel('Proportion of Samples')
        plt.ylabel('Cumulative Positive Rate')
        plt.title('Cumulative Accuracy Profile (CAP) Curve')
        plt.legend(loc='lower right')

        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided otherwise show plot
        if storage_path:
            plt.savefig(storage_path)
        else:
            plt.show()
