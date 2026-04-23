import json
import time
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ---------------------------
# LOGGING SETUP
# ---------------------------
def setup_logging():
    logger.add("logs/app.log", rotation="1 MB")
    logger.info("Logging initialized")


# ---------------------------
# SAVE RESULTS
# ---------------------------
def save_results(data, filename="results.json"):
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.info(f"Results saved to {file_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


# ---------------------------
# WORDCLOUD
# ---------------------------
def generate_wordcloud(texts, title="WordCloud"):
    try:
        text = " ".join(texts)

        wc = WordCloud(width=800, height=400, background_color="white").generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        plt.savefig(output_dir / "wordcloud.png")
        plt.close()

        logger.info("Word cloud generated")

    except Exception as e:
        logger.error(f"Wordcloud error: {e}")


# ---------------------------
# TIMER (OPTIONAL)
# ---------------------------
class Timer:
    def __init__(self, name="Task"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        logger.info(f"{self.name} took {end - self.start:.2f} seconds")