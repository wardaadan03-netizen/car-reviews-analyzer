from typing import List, Dict, Any
import pandas as pd
from config import config


class ReviewSummarizer:
    def __init__(self, llm_analyzer=None):
        self.llm_analyzer = llm_analyzer

    # -----------------------------
    # Main Review Summary
    # -----------------------------
    def generate_review_summary(self, reviews: List[str], max_length: int = 150) -> str:
        """Generate summary of reviews (LLM or fallback)"""

        # 🔴 If LLM not available → fallback
        if not self.llm_analyzer:
            return "LLM disabled. Simple summary: mostly mixed reviews with general opinions on performance, comfort, and fuel efficiency."

        combined_reviews = " ".join(reviews[:10])

        prompt = f"""
Summarize these car reviews in {max_length} words.

Focus on:
- positives
- negatives
- overall opinion

Reviews:
{combined_reviews}
"""

        try:
            if self.llm_analyzer.model_type == "openai":
                response = self.llm_analyzer.client.chat.completions.create(
                    model=self.llm_analyzer.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=max_length * 2
                )
                return response.choices[0].message.content

            else:
                response = self.llm_analyzer.client.messages.create(
                    model=self.llm_analyzer.model,
                    max_tokens=max_length * 2,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

        except Exception as e:
            return f"Summary failed: {str(e)}"

    # -----------------------------
    # Aspect Summary
    # -----------------------------
    def generate_aspect_summary(self, aspect_scores: pd.DataFrame) -> Dict[str, Any]:
        summary = {
            "best_aspects": [],
            "worst_aspects": [],
            "average_scores": {},
            "recommendations": []
        }

        for aspect in config.CAR_ASPECTS:
            if aspect in aspect_scores.columns:
                avg = aspect_scores[aspect].mean()
                summary["average_scores"][aspect] = avg

                if avg > 0.5:
                    summary["best_aspects"].append(aspect)
                elif avg < -0.3:
                    summary["worst_aspects"].append(aspect)

        sorted_aspects = sorted(summary["average_scores"].items(), key=lambda x: x[1], reverse=True)

        if sorted_aspects:
            summary["recommendations"].append(
                f"Strongest: {sorted_aspects[0][0].replace('_', ' ').title()}"
            )
            summary["recommendations"].append(
                f"Weakest: {sorted_aspects[-1][0].replace('_', ' ').title()}"
            )

        return summary

    # -----------------------------
    # Manufacturer Summary
    # -----------------------------
    def generate_manufacturer_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = df.copy()
        df["manufacturer"] = df["car_model"].astype(str).str.split().str[0]

        result = {}

        for m in df["manufacturer"].unique():
            group = df[df["manufacturer"] == m]

            result[m] = {
                "avg_rating": group["rating"].mean(),
                "num_reviews": len(group),
                "common_models": group["car_model"].value_counts().head(3).to_dict()
            }

            if "polarity" in df.columns:
                result[m]["avg_polarity"] = group["polarity"].mean()

        return result