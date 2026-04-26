"""
Fake News Detection — NLP Feature Extraction Module
Uses linguistic pattern analysis to pre-score articles before LLM analysis.
SDG 16 — Peace, Justice & Strong Institutions | SEAI Project
"""

import re
import logging

logger = logging.getLogger(__name__)

# Misinformation linguistic markers (based on academic misinformation research)
FAKE_MARKERS = [
    r"\bshocking\b",
    r"\byou won'?t believe\b",
    r"\bdoctors hate\b",
    r"\bsecret(?:ly)?\b",
    r"\bhidden from you\b",
    r"\bcover.?up\b",
    r"\bconspiracy\b",
    r"\bthey don'?t want you to know\b",
    r"\bmiracle cure\b",
    r"\bshare before\b",
    r"\bdelete(?:d)? it\b",
    r"\bwake up\b",
    r"\bexposed\b",
    r"\bplandemic\b",
    r"\bmicrochip\b",
    r"\bdeep state\b",
    r"\bfalse flag\b",
    r"\bhoax\b",
    r"\bthey lied\b",
    r"\bbig pharma\b",
    r"\bthey are hiding\b",
    r"!\s*!\s*!",          # Triple exclamation marks
    r"\b[A-Z]{6,}\b",      # SHOUTING words (6+ caps) — excludes normal acronyms like COVID/NASA/WHO
]

# Credibility markers from established journalism
CREDIBLE_MARKERS = [
    r"\baccording to\b",
    r"\bofficial(?:ly)?\b",
    r"\breport(?:ed|s|ing)?\b",
    r"\bstud(?:y|ies)\b",
    r"\bresearch(?:ers?)?\b",
    r"\buniversity\b",
    r"\bgovernment\b",
    r"\bstatement\b",
    r"\bconference\b",
    r"\banalyst(?:s)?\b",
    r"\bsource(?:s)?\b",
    r"\bdata\b",
    r"\bpercent(?:age)?\b",
    r"\bmillion\b",
    r"\bbillion\b",
    r"\bguideline(?:s)?\b",
    r"\brecommend(?:s|ed|ation)?\b",
    r"\bscientist(?:s)?\b",
    r"\bexpert(?:s)?\b",
    r"\bauthorit(?:y|ies)\b",
    r"\bminister\b",
    r"\bparliament\b",
    r"\bcourt\b",
    r"\btrial\b",
    r"\bpublish(?:ed)?\b",
    r"\bjournal\b",
]


def classify_news(text: str) -> dict:
    """
    Extract NLP features and compute a credibility pre-score.
    Returns verdict (REAL/FAKE), confidence %, and diagnostic info.
    """
    words = text.split()
    word_count = max(len(words), 1)

    # Count pattern matches (case-insensitive)
    fake_hits = sum(
        1 for p in FAKE_MARKERS if re.search(p, text, re.IGNORECASE)
    )
    credible_hits = sum(
        1 for p in CREDIBLE_MARKERS if re.search(p, text, re.IGNORECASE)
    )

    # Punctuation & style features
    exclamation_count = text.count('!')
    question_count = text.count('?')

    # ALL-CAPS ratio (only count words that are entirely uppercase and >3 chars)
    # This avoids penalizing normal acronyms like WHO, UN, US, AI, etc.
    shouting_words = sum(
        1 for w in words
        if w.isalpha() and w.isupper() and len(w) > 4
    )
    shouting_ratio = shouting_words / word_count

    # --- Score calculation (0 = fake, 100 = credible) ---
    base_score = 50.0
    base_score -= fake_hits * 10        # penalise each fake marker
    base_score += credible_hits * 7     # reward credibility markers
    base_score -= exclamation_count * 3 # penalise exclamation marks
    base_score -= shouting_ratio * 25   # penalise all-caps shouting

    # Bonus for neutral, factual tone (no exclamations, has credible hits)
    if exclamation_count == 0 and credible_hits >= 1:
        base_score += 5
    if credible_hits >= 3:
        base_score += 5

    base_score = max(0.0, min(100.0, base_score))

    verdict = 'REAL' if base_score >= 50 else 'FAKE'

    # Confidence = how far from neutral (50); min 52 so it's never exactly 50%
    distance = abs(base_score - 50)
    confidence = round(min(50 + distance + 2, 97.0), 2)

    logger.info(
        f"NLP score={base_score:.1f} | fake_hits={fake_hits} | "
        f"credible_hits={credible_hits} | excl={exclamation_count} | "
        f"shouting={shouting_words} | verdict={verdict} | conf={confidence}%"
    )

    return {
        'label': verdict,
        'score': confidence,
        'raw_score': base_score,
        'fake_hits': fake_hits,
        'credible_hits': credible_hits,
    }
