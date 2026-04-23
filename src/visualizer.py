import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path


class ReviewVisualizer:
    """Safe visualization module for car reviews"""

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Safe style fallback
        try:
            plt.style.use("ggplot")
        except:
            pass

        sns.set_theme(style="darkgrid")

    # -------------------------
    # Sentiment Distribution
    # -------------------------
    def plot_sentiment_distribution(self, df: pd.DataFrame, save=True):

        if "sentiment" not in df.columns:
            print("No sentiment column found")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        counts = df["sentiment"].value_counts()

        axes[0].bar(counts.index, counts.values)
        axes[0].set_title("Sentiment Distribution")

        if "polarity" in df.columns:
            axes[1].hist(df["polarity"], bins=15)
            axes[1].axvline(0, color="red")
            axes[1].set_title("Polarity Distribution")
        else:
            axes[1].text(0.5, 0.5, "No polarity data", ha="center")

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "sentiment.png")

        plt.show()

    # -------------------------
    # Rating vs Sentiment
    # -------------------------
    def plot_rating_vs_sentiment(self, df: pd.DataFrame, save=True):

        if "rating" not in df.columns:
            print("No rating column found")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if "polarity" in df.columns:
            axes[0].scatter(df["rating"], df["polarity"])
            axes[0].set_title("Rating vs Sentiment")

        if "polarity" in df.columns:
            df.boxplot(column="polarity", by="rating", ax=axes[1])
            axes[1].set_title("Polarity by Rating")

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "rating_sentiment.png")

        plt.show()

    # -------------------------
    # Radar Chart
    # -------------------------
    def plot_aspect_radar(self, aspect_scores: dict, title="Aspect Analysis", save=True):

        if not aspect_scores:
            print("No aspect data")
            return

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=list(aspect_scores.values()),
            theta=list(aspect_scores.keys()),
            fill="toself"
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(range=[-1, 1])),
            title=title
        )

        if save:
            fig.write_html(self.output_dir / "radar.html")

        fig.show()

    # -------------------------
    # Interactive Dashboard
    # -------------------------
    def plot_interactive_sentiment(self, df: pd.DataFrame, save=True):

        if df.empty:
            return

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Sentiment", "Trend", "Ratings", "Length"))

        if "sentiment" in df.columns:
            counts = df["sentiment"].value_counts()
            fig.add_trace(go.Bar(x=counts.index, y=counts.values), row=1, col=1)

        if "polarity" in df.columns:
            fig.add_trace(go.Scatter(y=df["polarity"]), row=1, col=2)

        if "rating" in df.columns:
            avg = df.groupby("car_model")["rating"].mean()
            fig.add_trace(go.Bar(x=avg.index, y=avg.values), row=2, col=1)

        if "review_length" in df.columns:
            fig.add_trace(go.Histogram(x=df["review_length"]), row=2, col=2)

        fig.update_layout(height=700, title="Dashboard")

        if save:
            fig.write_html(self.output_dir / "dashboard.html")

        fig.show()

    # -------------------------
    # Word Cloud Safe
    # -------------------------
    def plot_word_cloud_data(self, df: pd.DataFrame, column="review_text"):

        try:
            from wordcloud import WordCloud

            if column not in df.columns:
                print("No text column found")
                return

            text = " ".join(df[column].astype(str))

            wc = WordCloud(width=800, height=400).generate(text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title("Word Cloud")

            plt.savefig(self.output_dir / "wordcloud.png")
            plt.show()

        except ImportError:
            print("Install wordcloud: pip install wordcloud")