import pandas as pd

class TextPreprocessor:
    def preprocess_reviews(self, df):
        df = df.copy()
        df["review_text"] = df["review_text"].str.lower()
        return df