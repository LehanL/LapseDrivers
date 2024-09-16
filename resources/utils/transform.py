import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import skew

from resources.utils.sentiment import SentimentAnalyser

import logging
import logging.config

class DualSourceTransform:
    def __init__(self, df1, df2, keys, target_col, raw_text, how='inner'):
        """
        Initialise the Transform class to preprocess two source dataframes.

        This constructor sets up the merging operation between two supplied dataframes (`df1` and `df2`).
        The constructor also identifies columns to ignore during subsequent transformations.

        Parameters:
        - df1 (pd.DataFrame): The first source dataframe.
        - df2 (pd.DataFrame): The second source dataframe.
        - keys (str or list of str): The column(s) on which to perform the merge operation.
        - target_col (str): The target column to be used for modeling.
        - raw_text (str or list of str): The column containing raw text data to be used for processing.
        - how (str, optional): The type of merge operation to perform. Accepts 'inner', 'outer', 'left', or 'right'. Defaults to 'inner'.

        Attributes:
        - ignore_columns (set): Set of columns to ignore in transformations, including keys, target column, and raw text column.
        - merged_df (pd.DataFrame): Resulting dataframe after merging `df1` and `df2`.
        """
        self.logger = logging.getLogger('transform')
        
        self.df1 = df1
        self.df2 = df2
        self.keys = [keys] if isinstance(keys, str) else keys
        self.raw_text = [raw_text] if isinstance(raw_text, str) else raw_text
        self.target_col = target_col
        # create ignore columns list to ignore in transformations
        self.ignore_columns = set(self.keys + [self.target_col] + self.raw_text)
        self.how = how       
        
        # call merge dataframes function to join the two source tables
        self.merged_df =  self.merge_dataframes()

        
    def merge_dataframes(self):
        """
        Merge two DataFrames using one or more keys.

        Parameters:
        - df1: First DataFrame.
        - df2: Second DataFrame.
        - keys: A single key (string) or a list of keys (list of strings) for the merge.
        - how: Type of merge to perform ('left', 'right', 'outer', 'inner'). Default is 'inner'.

        Returns:
        - Merged DataFrame.
        """
        merged_df = pd.merge(self.df1, self.df2, how=self.how, on=self.keys)
        
        self.logger.info("Dataframes merged")
        
        return merged_df

    def drop_empty_keys(self):
        """
        Drop rows from a DataFrame where provided key variables (columns) are empty.

        Parameters:
        - self.df: The DataFrame from which to drop rows.
        - self.keys: A single key (str) or a list of keys (list of strings) to check for empty values.

        Returns:
        - DataFrame with rows dropped where any of the key variables are empty.
        """

        # Drop rows where any of the specified key variables are empty
        self.merged_df.dropna(subset=self.keys, inplace=True)
        
        self.logger.info("Rows with empty keys dropped")

    
    def drop_sparse_data(self, threshold=0.75):
        """
        Remove rows and columns from the DataFrame where threshold percentage or more of the values are missing.
        
        Parameters:
        - threshold (int): Proportion of missing values to determine if a row/column should be removed.
        
        Return: Alters self.merged_df pandas DataFrame.
        """
        # Calculate the threshold for missing rows within a column
        column_threshold = threshold * len(self.merged_df)
        
        # Identify columns to be dropped based on missing values
        columns_to_drop = [
            col for col in self.merged_df.columns
            if col not in self.ignore_columns and self.merged_df[col].isna().sum() > column_threshold
        ]

        # Remove columns with missing values above the threshold
        self.merged_df.drop(columns=columns_to_drop, inplace=True)

        # Calculate the threshold for missing columns within a row
        row_threshold = threshold * (len(self.merged_df.columns))
        
        # Remove rows with missing values above the threshold
        self.merged_df.dropna(axis=0, thresh=row_threshold, inplace=True)

        self.logger.info("Sparse rows and columns dropped")


    def apply_sentiment_analyser(self):
        """
        Apply sentiment analysis to each column specified in the `raw_text` list.

        Attributes:
        - self.raw_text (list of str): List of column names in the `merged_df` DataFrame that contain raw text data.
        
        Returns:
        - None: This method does not return a value but modifies the `merged_df` DataFrame.
        """
        
        # generate sentiment for each raw text column. 
        for text_col in self.raw_text:
            # initialise sentiment analyser
            sentiment_analyser = SentimentAnalyser(self.merged_df,text_col)
            self.merged_df = sentiment_analyser.apply_sentiment_scores()

            self.logger.info("Sentiment scores calculated and added to dataframe successfully.")

    
    
    def impute(self, segment_var=None, bins=5):
        """
        Impute missing values in the DataFrame. 
        Categorical columns missing values are replaced with 'unknown'.
        Numeric column missing values are replaced by the column mode or by the mean if there is no mode available. 
        If a `segment_var` is supplied the numeric column imputation is enhanced to replace missing values with the segment's column mode or mean if no mode is available.
        If `segment_var` is numeric, it bins the column and uses these bins for mode-based imputation. 

        Parameters:
        - segment_var (str or None): Column name for segmentation; if None, no segment-based numeric imputation.
        - bins (int): Number of bins for segmentation if `segment_var` is numeric.

        Returns:
        - None: Modifies `self.merged_df`.
        """

        # Bin the segmentation variable if it's numeric
        if segment_var and pd.api.types.is_numeric_dtype(self.merged_df[segment_var]):
            self.merged_df[segment_var + '_binned'] = pd.qcut(self.merged_df[segment_var], q=bins, labels=False, duplicates='drop')
            # default empty segment values with the bottom bin
            self.merged_df[segment_var + '_binned'] = self.merged_df[segment_var + '_binned'].fillna(0)
            # repoint segment_var to the binned variant
            segment_var = segment_var + '_binned'

        elif pd.api.types.is_string_dtype(self.merged_df[segment_var]):
            # if the segment var is 
            self.merged_df.fillna({segment_var: 'unknown'}, inplace=True)

        # Step through dataframe columns
        for col in self.merged_df.columns:
            if col in self.ignore_columns:
                # Ignore keys and target variable
                continue

            if self.merged_df[col].dtype in ['object', 'string', 'category']:
                # Impute categorical columns with 'unknown'
                self.merged_df.fillna({col: 'unknown'}, inplace=True)

            elif self.merged_df[col].dtype in ['int64', 'Int64', 'float64']:
                if segment_var is not None:
                # Impute numeric columns with the mode based on the segmentation variable if it exists
                    if pd.notna(self.merged_df[segment_var]).all():  # Ensure there are no NaNs in binned variable
                            mode_imputation = self.merged_df.groupby(segment_var)[col].apply(
                            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.mean()))
                            self.merged_df[col] = mode_imputation.reset_index(level=0, drop=True)
                else:
                    # Impute numeric columns with the overall mode or mean if no segment_var is provided
                    mode_value = self.merged_df[col].mode()[0] if not self.merged_df[col].mode().empty else self.merged_df[col].mean()
                    self.merged_df.fillna({col: mode_value}, inplace=True)
        
        # drop binned column if one was created for imputation
        if '_binned' in segment_var:
            self.merged_df.drop(columns=[segment_var], inplace=True)

        self.logger.info("Imputation successful")

    def encoder(self): 
        """
        Encodes specified columns in the DataFrame using one-hot encoding.

        Returns:
        - None: Modifies `self.merged_df`.
        """
        # Initialise list to collect columns to be dropped later
        columns_to_drop = []

        # List of columns that were already one-hot encoded
        one_hot_encoded_columns = []

        # Iterate through DataFrame columns
        for col in self.merged_df.columns:
            if col in self.ignore_columns:
                # Skip columns in the ignore list
                continue

            # Check if the column is of string type
            if pd.api.types.is_string_dtype(self.merged_df[col]):
                # Standardise text columns by converting to lowercase
                self.merged_df[col] = self.merged_df[col].str.strip().str.lower()

                # One-hot encode the column
                one_hot = pd.get_dummies(self.merged_df[col], prefix=col)

                # Convert one-hot encoded columns to integers
                one_hot = one_hot.astype(int)
                
                # Drop the original column and add one-hot encoded columns
                self.merged_df = self.merged_df.drop(col, axis=1)
                self.merged_df = pd.concat([self.merged_df, one_hot], axis=1)
            
        
        self.logger.info("One-hot encoding successful") 

    def scaler(self, log_transform_threshold=0.5): 
        """
        Apply Min-Max scaling to numeric columns in the DataFrame.
        The function first assesses the numeric columns skewness, if above the set threshold log transformation will be applied before min-max scaling.

        Parameters:
        - log_transform_threshold (float, optional): The threshold for skewness above which log transformation is applied. Default is 0.5.

        Returns:
        - None: Modifies `self.merged_df` in place.
        """
        # Initialise MinMaxScaler
        scaler = MinMaxScaler()

        # Iterate over columns to preprocess
        for col in  self.merged_df.columns:
            if col in self.ignore_columns:
                # Skip columns in the ignore list
                continue
            
            # identify numeric columns
            if pd.api.types.is_numeric_dtype(self.merged_df[col]) and not pd.api.types.is_bool_dtype(self.merged_df[col]):
            # Check skewness to determine if log transformation is needed
                if (self.merged_df[col] >= 0).all() and abs(skew(self.merged_df[col].dropna())) > log_transform_threshold:
                    # Apply log transformation
                    self.merged_df[col] = np.log1p(self.merged_df[col])  # Clip lower bound to avoid log(0)
                    
                # Apply Min-Max Scaling
                self.merged_df[[col]] = scaler.fit_transform(self.merged_df[[col]])

        self.logger.info("Numerical min-max scaling successful")


    def drop_column(self,drop_col):
        """
        Drop specified column within the merged DataFrame.

        Returns:
        - None: Modifies `self.merged_df` in place.
        """
        # Drop supplied column
        self.merged_df.drop(columns=[drop_col], inplace=True)

        self.logger.info(f"Column {drop_col} successfully dropped")


    def get_transformed_df(self):
        """
        Return the merged DataFrame.

        Returns:
        - Merged Pandas DataFrame.
        """
        return self.merged_df
    
    def store_data(self, save_path):
        """
        Save the processed data as CSV.

        Parameters:
        - save_path (str): OS path where dataframe will be stored as .csv

        Returns:
        - None: Stores Merged Pandas Dataframe in path provided.
        """
        try:
            # Save the DataFrame as a CSV file
            self.merged_df.to_csv(save_path, index=False)

            self.logger.info(f"Processed data successfully saved to {save_path}")

        except Exception as e:
            self.logger.error(e)
    

    def split_by_date(self, date_col, split_date):
        """
        Split `self.merged_df` into two DataFrames based on the provided datetime.

        Parameters:
        - date_col (str): The name of the column containing datetime values used for splitting.
        - split_date (datetime-like): The datetime value used to split the DataFrame.

        Returns:
        - score_df (pd.DataFrame): DataFrame with rows on or after `split_date`.
        """
        # Ensure the date column is in datetime format
        self.merged_df[date_col] = pd.to_datetime(self.merged_df[date_col], errors='coerce')

        # Split the DataFrame based on the provided split_date
        score_df = self.merged_df[self.merged_df[date_col] >= split_date]
        self.merged_df = self.merged_df[self.merged_df[date_col] < split_date]
        

        self.logger.info(f"Scoring set split from self.merged_df by date: {split_date}")

        return score_df

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.

        Parameters:
        - test_size (float): Proportion of the dataset to include in the test split.
        - random_state (int): Seed used by the random number generator.

        Returns:
        - x_train (pd.DataFrame): Pandas Dataframe for machine learning training.
        - x_test (pd.DataFrame): Pandas Dataframe for model validation.
        - y_train (pd.DataFrame): Pandas Dataframe containing the event field for machine learning training.
        - y_test (pd.DataFrame): Pandas Dataframe containing the event field for model validation.
        - features (list of str): List containing the features considered for machine learning.
        """
        
        features = [col for col in self.merged_df.columns if col not in self.ignore_columns]

        x_train, x_test, y_train, y_test = train_test_split(
            self.merged_df[features], self.merged_df[self.target_col], test_size=test_size, random_state=random_state)
        
        self.logger.info("Train Test split performed successfully")
            
        return  x_train, x_test, y_train, y_test, features


            
        
        
    
