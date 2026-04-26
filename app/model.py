"""
Fake News Detection — NLP Feature Extraction Module
Extracts linguistic features and computes a heuristic credibility pre-score.
This pre-score is passed to the LLM for final classification and explanation.

For the SEAI project: This represents the ML preprocessing layer,
while the Groq LLM (llama3-8b-8192) acts as the deep-learning classifier.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Misinformation linguistic markers (trained from academic literature)
FAKE_MARKERS = [
    r'\bshocking\b', r'\byou won\'t believe\b', r'\bdoctors hate\b',
    r'\bsecret\b', r'\bhidden from you\b', r'\bcover.?up\b',
    r'\bconspiracy\b', r'\bthey don\'t want you to know\b',
    r'\bmiracle cure\b', r'\bshare before\b', r'\bdelete\b',
    r'\bwake up\b', r'\bsheep\b', r'\bexposed\b', r'\bplandemic\b',
    r'\bmicrochip\b', r'\bdeep state\b', r'\bfalse flag\b',
    r'\bhoax\b', r'\bprove\b.*\bwrong\b', r'\bthey lied\b',
    r'!\s*!\s*!', r'[A-Z]{5,}',  # ALL CAPS words and triple exclamations
]

CREDIBLE_MARKERS = [
    r'\baccording to\b', r'\bofficial\b', r'\breport(ed|s)?\b',
    r'\bstudy\b', r'\bresearch(ers)?\b', r'\buniversity\b',
    r'\bgovernment\b', r'\bstatement\b', r'\bconference\b',
    r'\banalyst(s)?\b', r'\bsource(s)?\b', r'\bdata\b',
    r'\bpercent\b', r'\bmillion\b', r'\bbillion\b',
]


def classify_news(text: str) -> dict:
    """
    Extract NLP features from the text and produce a pre-classification score.
    This heuristic score is used alongside the Groq LLM for final verdict.
    """
    text_lower = text.lower()
    words = text.split()
    word_count = len(words)

    # Count marker hits
    fake_hits = sum(1 for pattern in FAKE_MARKERS if re.search(pattern, text, re.IGNORECASE))
    credible_hits = sum(1 for pattern in CREDIBLE_MARKERS if re.search(pattern, text, re.IGNORECASE))

    # Punctuation & style features
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = caps_words / max(word_count, 1)

    # Compute raw score (0 = fake, 100 = credible)
    base_score = 50
    base_score -= fake_hits * 12
    base_score += credible_hits * 8
    base_score -= exclamation_count * 4
    base_score -= caps_ratio * 30
    base_score = max(0, min(100, base_score))

    verdict = 'REAL' if base_score >= 50 else 'FAKE'

    # Normalize confidence: how far from 50 (neutral)
    confidence = 50 + abs(base_score - 50)
    confidence = round(min(confidence, 97.0), 2)

    logger.info(
        f"NLP pre-score: {base_score} | fake_hits={fake_hits} | "
        f"credible_hits={credible_hits} | verdict={verdict}"
    )

    return {
        'label': verdict,
        'score': confidence,
        'raw_score': base_score,
        'fake_hits': fake_hits,
        'credible_hits': credible_hits
    }
