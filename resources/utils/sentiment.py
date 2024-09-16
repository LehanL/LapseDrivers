import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

class SentimentAnalyser:
    def __init__(self, df, text_column):
        """
        Initialises the SentimentAnalyser class.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the text data.
        - text_column (str): The name of the column containing raw text data.
        """

        self.df = df
        self.text_column = text_column
        self.sentiment_analyser = SentimentIntensityAnalyzer()  # Initialise VADER sentiment analyser

    def calculate_sentiment(self, text):
        """
        Calculates the sentiment score for a given text using VADER.

        Parameters:
        - text (str): The text to score.

        Returns:
        - float: Sentiment score.
        """
        if pd.isna(text):
            # Return a neutral score for missing text
            return 0.0
        
        # Convert to string if not already
        text = str(text)  
        
        # Use VADER to compute sentiment scores
        sentiment_score = self.sentiment_analyser.polarity_scores(text)
        
        return sentiment_score['compound']  # Return the compound score

    def apply_sentiment_scores(self):
        """
        Applies sentiment scores to the DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with a new column containing sentiment scores.
        """
        if isinstance(self.text_column, str) and self.text_column in self.df.columns:
            # Apply the sentiment score calculation to each row in the DataFrame
            self.df['sentiment_score'] = self.df[self.text_column].apply(self.calculate_sentiment)
        else:
            raise ValueError("text_column must be a valid string column name in the DataFrame.")
        return self.df
