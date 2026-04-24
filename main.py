import click
import pandas as pd
import json

from src.data_loader import CarReviewsLoader
import click
import pandas as pd
import json

from src.data_loader import CarReviewsLoader
from src.preprocessor import TextPreprocessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.feature_extractor import FeatureExtractor
from src.summarizer import ReviewSummarizer
from src.visualizer import ReviewVisualizer
from src.llm_analyzer import LLMAnalyzer
from config import config


class CarReviewAnalyzer:
    def __init__(self, use_llm=True, llm_model="openai"):
        self.loader = CarReviewsLoader()
        self.preprocessor = TextPreprocessor()
        self.sentiment = SentimentAnalyzer()
        self.feature_extractor = FeatureExtractor()

        # LLM optional
        self.llm_analyzer = LLMAnalyzer(llm_model) if use_llm else None
        self.summarizer = ReviewSummarizer(self.llm_analyzer)
        self.visualizer = ReviewVisualizer()

        config.setup_directories()

    def run_analysis(self, data_source="sample", use_llm=True):
        print("🚗 Running Car Review Analysis...\n")

        # 1. Load data
        if data_source == "sample":
            df = self.loader.load_sample_data()
        else:
            df = self.loader.load_from_csv(data_source)

        print(f"Loaded {len(df)} reviews")

        # 2. Preprocess
        df = self.preprocessor.preprocess_reviews(df)

        # 3. Sentiment analysis
        df = self.sentiment.analyze_reviews(df)

        # 4. Feature extraction (FIXED)
        df = self.feature_extractor.extract_rating_features(df)
        df = self.feature_extractor.extract_text_features(df)

        # 5. LLM (SAFE)
        if use_llm and self.llm_analyzer:
            print("🤖 Running LLM analysis...")
            reviews = df["review_text"].tolist()[:3]
            llm_results = self.llm_analyzer.batch_analyze(reviews, "sentiment")
            print("LLM Results:", llm_results)

        # 6. Summary
        summary = self.summarizer.generate_review_summary(df["review_text"].tolist())
        print("\n📄 Summary:", summary)

        # 7. Visualization
        self.visualizer.plot_sentiment_distribution(df)

        # 8. Save
        self.loader.save_processed_data(df, "output.csv")

        print("\n✅ Done!")
        return df


@click.command()
@click.option("--data-source", default="sample")
@click.option("--use-llm/--no-llm", default=True)
@click.option("--llm-model", default="openai")
def main(data_source, use_llm, llm_model):
    analyzer = CarReviewAnalyzer(use_llm, llm_model)
    analyzer.run_analysis(data_source, use_llm)


if __name__ == "__main__":
    main()