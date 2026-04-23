import unittest
import pandas as pd
from src.data_loader import CarReviewsLoader
from src.preprocessor import TextPreprocessor
from src.sentiment_analyzer import SentimentAnalyzer

class TestCarReviewAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.loader = CarReviewsLoader()
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def test_data_loading(self):
        df = self.loader.load_sample_data()
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn('review_text', df.columns)
        
    def test_preprocessing(self):
        text = "The car is AMAZING!!!"
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "the car is amazing")
        
    def test_sentiment_analysis(self):
        df = self.loader.load_sample_data()
        df = self.sentiment_analyzer.analyze_reviews(df)
        self.assertIn('sentiment', df.columns)
        self.assertIn('polarity', df.columns)
        
    def test_review_length(self):
        df = self.loader.load_sample_data()
        df = self.preprocessor.preprocess_reviews(df)
        self.assertIn('review_length', df.columns)
        self.assertGreater(df['review_length'].min(), 0)

if __name__ == '__main__':
    unittest.main()