import pandas as pd
import json
from config import config


class CarReviewsLoader:
    def load_sample_data(self):
        return pd.DataFrame({
            "review_text": [
                "Great car, very smooth ride",
                "Bad mileage and poor build quality",
                "Comfortable and safe vehicle",
                "Engine is powerful but noisy",
                "Excellent value for money"
            ],
            "car_model": [
                "Tesla Model 3",
                "Toyota Prius",
                "Honda CR-V",
                "BMW X3",
                "Hyundai Elantra"
            ],
            "rating": [5, 2, 4, 3, 5]
        })

    def load_from_csv(self, path):
        return pd.read_csv(path)

    def save_processed_data(self, df, filename):
        path = config.PROCESSED_DATA_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)