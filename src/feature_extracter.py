import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """Feature extraction for car reviews"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

    # ---------------------------
    # Basic Numeric Features
    # ---------------------------
    def extract_rating_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize rating (0–1 scale)
        df['rating_scaled'] = df['rating'] / 5.0
        return df

    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Review length (number of words)
        df['review_length'] = df['review_text'].apply(lambda x: len(str(x).split()))
        return df

    # ---------------------------
    # TF-IDF Features (Text → Numbers)
    # ---------------------------
    def extract_tfidf_features(self, texts):
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        # Convert to DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        return tfidf_df

    # ---------------------------
    # Key Terms Extraction
    # ---------------------------
    def extract_key_terms(self, df: pd.DataFrame):
        key_terms = {}

        for sentiment in df['sentiment'].unique():
            subset = df[df['sentiment'] == sentiment]

            text = " ".join(subset['review_text'].astype(str))

            tfidf = TfidfVectorizer(max_features=10, stop_words='english')
            tfidf_matrix = tfidf.fit_transform([text])

            terms = dict(zip(tfidf.get_feature_names_out(), tfidf_matrix.toarray()[0]))
            key_terms[sentiment] = terms

        return key_terms