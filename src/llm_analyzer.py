from openai import OpenAI
from anthropic import Anthropic
from typing import List, Dict, Any
import json
import os
import re
from config import config


class LLMAnalyzer:
    """Robust LLM analysis for car reviews"""

    def __init__(self, model_type: str = "openai"):
        self.model_type = model_type

        if model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")

            self.client = OpenAI(api_key=api_key)
            self.model = getattr(config, "DEFAULT_MODEL", "gpt-4o-mini")

        elif model_type == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-haiku-20240307"

        else:
            raise ValueError("model_type must be 'openai' or 'anthropic'")

    # ---------------------------
    # Sentiment Analysis
    # ---------------------------
    def analyze_sentiment(self, review: str) -> Dict[str, Any]:
        prompt = f"""
Return ONLY valid JSON.

Analyze sentiment:
- sentiment: positive/negative/neutral
- confidence: 0-1
- key_phrases: list

Review: {review}
"""

        response = self._generate(prompt)
        return self._safe_json(response)

    # ---------------------------
    # Aspect Extraction
    # ---------------------------
    def extract_aspects(self, review: str) -> Dict[str, float]:
        aspects = getattr(config, "CAR_ASPECTS", [])

        prompt = f"""
Return ONLY valid JSON.

Rate aspects (-1 to 1):

Aspects: {", ".join(aspects)}

Review: {review}
"""

        response = self._generate(prompt)
        result = self._safe_json(response)

        for a in aspects:
            result.setdefault(a, 0.0)

        return result

    # ---------------------------
    # Batch
    # ---------------------------
    def batch_analyze(self, reviews: List[str], analysis_type: str = "sentiment"):
        results = []

        for r in reviews:
            try:
                if analysis_type == "sentiment":
                    results.append(self.analyze_sentiment(r))
                elif analysis_type == "aspects":
                    results.append(self.extract_aspects(r))
                else:
                    results.append({"error": "invalid type"})
            except Exception as e:
                results.append({"error": str(e)})

        return results

    # ---------------------------
    # API CALL
    # ---------------------------
    def _generate(self, prompt: str) -> str:
        try:
            if self.model_type == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()

            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ---------------------------
    # SAFE JSON PARSER (FIXED)
    # ---------------------------
    def _safe_json(self, text: str) -> Dict:
        try:
            # remove markdown blocks if any
            text = re.sub(r"```json|```", "", text).strip()

            # extract JSON part if extra text exists
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())

            return json.loads(text)

        except Exception:
            return {
                "error": "Invalid JSON",
                "raw": text
            }
            