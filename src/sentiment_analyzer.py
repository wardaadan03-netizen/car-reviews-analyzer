from textblob import TextBlob
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

class SentimentAnalyzer:
    def analyze_reviews(self, df):
        df = df.copy()

        def simple_sentiment(text):
            if "bad" in text or "poor" in text:
                return "negative"
            elif "excellent" in text or "great" in text:
                return "positive"
            return "neutral"

        df["sentiment"] = df["review_text"].apply(simple_sentiment)
        return df