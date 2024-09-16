from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import pickle

from resources.utils.visualise import ModelPerformancePlotter

import logging
import logging.config


class HistGBMPipeline:
    def __init__(self, x_train, x_test, y_train, y_test, features, target_col):
        """
        Initialises the HistGBM modeling pipeline.

         Parameters:
        - x_train (pd.DataFrame): The DataFrame containing the training features.
        - x_test (pd.DataFrame): The DataFrame containing the testing features.
        - y_train (pd.Series or pd.DataFrame): The target variable for training data.
        - y_test (pd.Series or pd.DataFrame): The target variable for testing data.
        - features (list or str): A list of feature names to be used in the model. 
          If a single feature name is provided as a string, it will be converted to a list.
        - target_col (str): The name of the column representing the target variable.

        Attributes:
        - model (HistGradientBoostingClassifier): Placeholder for the HistGBM model instance.
        - y_pred (pd.Series or np.ndarray): Placeholder for predicted target values.
        - y_prob (pd.Series or np.ndarray): Placeholder for predicted probabilities (if applicable).

        """
        self.logger = logging.getLogger('model')

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = [features] if isinstance(features, str) else features
        self.target_col = target_col
        
        # Placeholder for model and data
        self.model = None
        self.y_pred = None 
        self.y_prob = None 
        self.score_dict = {}
        

    def HistGBM_optimise_and_train(self):
        """
        Optimises hyperparameters for the HistGradientBoostingClassifier using RandomizedSearchCV 
        and trains the model with the best parameters.

        Returns:
        - None: Allocates HistGBM model trained on the optimum hyperparameters to self.model
        """
        # Define the base model
        base_model = HistGradientBoostingClassifier()

        # Define the parameter grid for RandomizedSearchCV
        param_distributions = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_iter': [100, 200, 300, 400, 500],
            'max_leaf_nodes': [15, 31, 63, 127],
            'max_depth': [3, 5, 7, 9, None],
            'min_samples_leaf': [10, 20, 30, 40, 50],
            'l2_regularization': [0.0, 0.1, 0.5, 1.0],
            'early_stopping': [True],
            'scoring': ['roc_auc'],
            'validation_fraction': [0.1, 0.15, 0.2]
        }

        # Setup the RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=50,  # Number of iterations to perform
            scoring='roc_auc',  # Optimise based on AUC score
            cv=5,  # 5-fold cross-validation
            verbose=1,  # Set to 1 or higher to get updates
            random_state=42,  # Seed for reproducibility
            n_jobs=-1  # Use all available cores
        )

        # Fit the RandomizedSearchCV
        random_search.fit(self.x_train, self.y_train)

        self.logger.info("Randomised search hyperparameter optimisation complete")

        # Extract the best model
        best_params = random_search.best_params_
        best_params['early_stopping'] = False  # Disable early stopping for final model

        # Reinitialise and fit the final model on the full training data with best parameters
        self.model = HistGradientBoostingClassifier(**best_params)
        self.model.fit(self.x_train, self.y_train)

        self.logger.info(f"HistGBM model has been refit with the best hyperparameters found: {random_search.best_params_}")

    def validate_performance(self):
        """
        Validates the performance of the trained model by computing various metrics.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, specificity, 
                AUC (Area Under the ROC Curve), and Gini coefficient.

        Raises:
            ValueError: If the model has not been trained.
        """
        # Check if model has been trained
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # use model to predict target for the test set
        self.y_pred = self.model.predict(self.x_test)
        
        # use model to predict target probabilities for the test set
        self.y_prob = self.model.predict_proba(self.x_test)[:, 1]

        # Compute performance metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()

        # Compute specificity
        specificity = tn / (tn + fp)

        # Compute auc score and gini
        auc_score = roc_auc_score(self.y_test, self.y_prob)
        gini_score = 2 * auc_score - 1

        # compile performance metrics
        self.score_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity':specificity,
            'auc': auc_score,
            'gini': gini_score
        }

        self.logger.info(f"HistGBM model validation performance: {self.score_dict}")


    def plot_performance(self, storage_path=None):
        """
        Plots performance metrics. Results are stored to the provided storage_path.
        """
        # run validate_perfomance if performance metrics have not yet been generated.
        if not self.score_dict:
            self.score_dict = self.validate_performance()

        # initialise performance plotter
        plotter = ModelPerformancePlotter(self.y_pred, self.y_prob, self.y_test, self.score_dict)
        # plot performance metrics providing it the storage path
        plotter.plot_performance(storage_path)

        self.logger.info(f"Graph plotted and stored in {storage_path}")


    def save_model(self, filepath):
        """
        Saves the trained model to a file using pickle.

        Parameters:
        - filepath (str): The path where the model should be saved.
        """
        # check if model exists
        if self.model is None:
            self.logger.warning("No model is trained to save.")
            return
        
        # save model
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

        self.logger.info(f"Model saved to {filepath}.")
        

    def load_model(self, filepath):
        """
        Loads a model from a file using pickle and assigns it to self.model.

        Parameters:
        - filepath (str): The path from where the model should be loaded.
        """
        # load model from provided path 
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"Model loaded from {filepath}.")

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            self.logger.error(e)

    def get_model(self):
        """
        Returns the trained model.

        Returns:
            model: The trained model instance.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        # check if model exists
        if self.model is None:
            self.logger.warning("No model is trained or loaded to return.")
            return
        
        return self.model

    def score(self, x_score, return_proba=False):
        """
        Scores the provided dataset using the trained model.

        Args:
            x_score (pd.DataFrame or np.ndarray): Dataframe to be scoring.
            return_proba (bool): If True, returns probabilities; otherwise, returns predicted labels.

        Returns:
            np.ndarray: Predicted labels or probabilities for the input dataset.

        Raises:
            ValueError: If the model has not been trained.
        """
        # check if model exists
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Predict labels or probabilities
        if return_proba:
            y_scored = self.model.predict_proba(x_score)[:, 1]  # Probability for the positive class
        else:
            y_scored = self.model.predict(x_score)

        return y_scored
    



