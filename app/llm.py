"""
LLM Integration: Groq API with LLaMA 3
Generates detailed fact-check explanations for news articles.
API key loaded from .env file or GROQ_API_KEY environment variable.
"""

from groq import Groq
import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Resolve .env from the project root (parent of the app/ folder)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

logger = logging.getLogger(__name__)

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def explain_with_llm(text: str, classification: dict) -> dict:
    """
    Use Groq LLM to generate a detailed fact-check explanation.
    Returns dict with analysis, red_flags, credibility_score, fact_check_summary.
    """
    verdict = classification['label']
    confidence = classification['score']

    # Re-load .env and read key fresh each call
    load_dotenv(dotenv_path=_env_path, override=True)
    groq_api_key = os.environ.get("GROQ_API_KEY", "")

    # --- Fallback if no API key ---
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not set — returning fallback explanation.")
        return _fallback_explanation(verdict, confidence)

    client = Groq(api_key=groq_api_key)

    prompt = f"""You are an expert investigative journalist and fact-checker specializing in misinformation detection.

An AI classification model analyzed the following news article/headline and classified it as: **{verdict}** with {confidence:.1f}% confidence.

Article:
\"\"\"{text}\"\"\"

Your task: Provide a thorough fact-check analysis. Consider:
- Writing style, emotional language, sensationalism
- Verifiable claims vs. unverifiable assertions
- Source credibility indicators
- Common misinformation patterns

Respond ONLY with a valid JSON object in this exact format (no markdown, no code blocks):
{{
    "analysis": "A detailed 3-4 sentence expert analysis explaining your findings about this article's credibility. Be specific about what you found.",
    "red_flags": ["specific red flag or positive indicator 1", "specific red flag or positive indicator 2", "specific red flag or positive indicator 3", "add more as needed"],
    "credibility_score": <integer 0-100, where 0=completely fabricated, 100=highly credible>,
    "fact_check_summary": "A single clear sentence verdict suitable for the general public."
}}"""

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional fact-checker. Always respond with valid JSON only, no extra text."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=900
        )

        response_text = completion.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        result = json.loads(response_text)

        # Validate required keys
        for key in ["analysis", "red_flags", "credibility_score", "fact_check_summary"]:
            if key not in result:
                raise ValueError(f"Missing key: {key}")

        if not isinstance(result["red_flags"], list):
            result["red_flags"] = [str(result["red_flags"])]

        result["credibility_score"] = max(0, min(100, int(result["credibility_score"])))
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from LLM: {e}")
        return _fallback_explanation(verdict, confidence)
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return _fallback_explanation(verdict, confidence)


def _fallback_explanation(verdict: str, confidence: float) -> dict:
    """Fallback when LLM is unavailable."""
    if verdict == "FAKE":
        return {
            "analysis": (
                f"The BERT model classified this as FAKE with {confidence:.1f}% confidence. "
                "Patterns commonly associated with misinformation were detected. "
                "Manual verification is strongly recommended before sharing."
            ),
            "red_flags": [
                "AI model flagged as potential misinformation",
                "High-confidence fake news classification",
                "Manual fact-checking recommended",
                "Add GROQ_API_KEY for detailed LLM analysis"
            ],
            "credibility_score": max(5, int(100 - confidence)),
            "fact_check_summary": "AI model classifies this as likely FAKE news. Verify before sharing."
        }
    else:
        return {
            "analysis": (
                f"The BERT model classified this as REAL with {confidence:.1f}% confidence. "
                "Patterns consistent with credible journalism were detected. "
                "Always verify claims through trusted sources before drawing conclusions."
            ),
            "red_flags": [
                "AI model classified as likely credible content",
                "Language patterns consistent with factual reporting",
                "Further verification through trusted sources recommended",
                "Add GROQ_API_KEY for detailed LLM analysis"
            ],
            "credibility_score": int(confidence),
            "fact_check_summary": "AI model classifies this as likely REAL news. Cross-verify with trusted sources."
        }
