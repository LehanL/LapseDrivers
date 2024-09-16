import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns

import logging
import logging.config

class FeatureImportanceAnalyser:
    def __init__(self, model, x_train, x_test, x_score, y_test, y_pred):
        """
        Initialises the ModelReport class.

        Parameters:
        - model (HistGBMClassifier): The trained HistGBM model.
        - x_train (pd.DataFrame): Training features.
        - x_test (pd.DataFrame): Test features.
        - x_score (pd.Series): Training target.
        - y_test (pd.Series): Test target.
        - y_pred (pd.Series): Predicted target.


        """
        self.logger = logging.getLogger('report')

        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.x_score = x_score
        self.y_test = y_test
        self.y_pred = y_pred

        # Placeholders for feature importance data
        self.permutation_importance = None
        self.shap_values = None
        self.shap_agg_values = None
        self.lime_importance = None

    def compute_permutation_importance(self):
        """Compute and log permutation importance."""
        result = permutation_importance(self.model, self.x_test, self.y_test, n_repeats=10, random_state=42)
        self.permutation_importance = result.importances_mean
        self.logger.info(f"Permutation Importance: {self.permutation_importance}")

    def compute_tree_shap_values(self):
        """Compute and log Tree SHAP values."""
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer(self.x_score)
        self.shap_agg_values = np.abs(self.shap_values.values).mean(axis=0)
        self.logger.info(f"Tree SHAP Values: {self.shap_values}")

    def shap_analysis(self, shap_path='.', max_display = 10):
        """
        Plot Tree SHAP values.
        
        Parameters:
        - shap_path (str): Path to save the SHAP summary plot.
        """
        plt.figure()
        shap.summary_plot(self.shap_values, self.x_score, max_display=max_display)
        plt.savefig(f"{shap_path}")
        plt.close()

        self.logger.info(f"Shap analysis saved to: {shap_path}")


    def compute_lime_importance(self):
        """Compute and log LIME importance."""
        explainer = lime.lime_tabular.LimeTabularExplainer(self.x_score.values, mode='classification', 
                                                           feature_names=self.x_score.columns.tolist(),
                                                           class_names=[str(c) for c in np.unique(self.y_pred)], 
                                                           # verbose=True, #Uncomment line if needed to debug lime operation
                                                           discretize_continuous=True)
        lime_importances = []
        for i in range(len(self.x_score)):
            explanation = explainer.explain_instance(self.x_score.iloc[i].values, self.model.predict_proba, num_features=len(self.x_score.columns))
            importance_dict = dict(explanation.as_list())
            lime_importances.append([importance_dict.get(feat, 0) for feat in self.x_score.columns])
        self.lime_importance = np.mean(lime_importances, axis=0)
        
        self.logger.info(f"LIME Importance: {self.lime_importance}")

    def lime_analysis(self, index=0, lime_path='.'):
        """
        Generate LIME analyses for a given index and save the results.

        Parameters:
        - index (int): The index of the instance to analyse.
        - lime_path (str): Path to save the LIME .html file.
        """
        explainer = lime.lime_tabular.LimeTabularExplainer(self.x_score.values, mode='classification', 
                                                           feature_names=self.x_score.columns.tolist(),
                                                           class_names=[str(c) for c in np.unique(self.y_pred)], 
                                                           # verbose=True, #Uncomment line if needed to debug lime operation
                                                           discretize_continuous=True)

        lime_exp = explainer.explain_instance(
            data_row=self.x_score.iloc[index].values,
            predict_fn=self.model.predict_proba
        )
        
        # Save LIME explanation to an HTML file
        lime_exp.save_to_file(lime_path)

        self.logger.info(f"Lime analysis of sample {index} saved to: {lime_path}")

    def generate_graphs(self, save_path='.'):
        """
        Generate and save graphs of feature importance scores.
        
        Parameters:
        - save_path (str): Path to save the feature importance graphs.
        """
        # initialise figure
        plt.figure(figsize=(16, 10))

        # Permutation Importance
        if self.permutation_importance is not None:
            plt.subplot(1, 3, 1)
            sns.barplot(x=self.permutation_importance, y=self.x_test.columns)
            plt.ylabel('Features')
            plt.title('Permutation Importance')

        # Tree SHAP Values
        if self.shap_values is not None:
            plt.subplot(1, 3, 2)
            sns.barplot(x=self.shap_agg_values, y=self.x_score.columns)
            plt.ylabel('Features')
            plt.title('Tree SHAP Values')

        # LIME Importance
        if self.lime_importance is not None:
            plt.subplot(1, 3, 3)
            sns.barplot(x=self.lime_importance, y=self.x_score.columns)
            plt.ylabel('Features')
            plt.title('LIME Importance')

        # set graph layout
        plt.tight_layout()
        # save figure
        plt.savefig(f"{save_path}")
        # show figure
        plt.show()

        self.logger.info(f"Graphs saved to {save_path}")


    def export_csv(self, save_path='.'):
        """
        Export all importance scores to a CSV file.
        
        Parameters:
        - save_path (str): Path to save the feature importance summary .csv.
        """
        # create Pandas dataframe with all of the aggregated feature importance values
        importance_df = pd.DataFrame({
            'Feature': self.x_train.columns,
            'Permutation Importance': self.permutation_importance if self.permutation_importance is not None else np.nan,
            'Tree SHAP Values': self.shap_agg_values if self.shap_agg_values is not None else np.nan,
            'LIME Importance': self.lime_importance if self.lime_importance is not None else np.nan
        })

        # store dataframe in a .csv
        importance_df.to_csv(f"{save_path}", index=False)

        self.logger.info(f"CSV file saved to {save_path}")
