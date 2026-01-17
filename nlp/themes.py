# nlp/themes.py
# Headline theme extraction using TF-IDF (no external APIs).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class ThemeResult:
    theme: str
    keywords: List[str]
    sample_titles: List[str]


def _clean_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_themes_from_headlines(
    headlines: List[Dict[str, Any]],
    *,
    max_themes: int = 5,
    keywords_per_theme: int = 4,
    samples_per_theme: int = 3,
) -> Dict[str, Any]:
    """
    Lightweight, explainable theme extraction:
      - TF-IDF on titles
      - Select top-weighted terms overall
      - Group terms into "themes" by simple chunking (viva-friendly)
    Returns:
      {
        "themes": [ {theme, keywords, sample_titles}, ... ],
        "note": "...",
      }
    """
    titles: List[str] = []
    raw_titles: List[str] = []

    for h in headlines or []:
        title = h.get("title") or ""
        if not title.strip():
            continue
        raw_titles.append(title.strip())
        titles.append(_clean_text(title))

    if len(titles) < 5:
        return {
            "themes": [],
            "note": "Not enough headlines to extract stable themes.",
        }

    # TF-IDF
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=250,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vec.fit_transform(titles)
    feature_names = vec.get_feature_names_out()

    # Get overall term importance
    importances = X.sum(axis=0).A1  # sum tf-idf across docs
    ranked_idx = importances.argsort()[::-1]
    ranked_terms = [feature_names[i] for i in ranked_idx if importances[i] > 0]

    if not ranked_terms:
        return {"themes": [], "note": "Could not compute meaningful TF-IDF themes."}

    # Build themes by chunking top terms
    top_terms = ranked_terms[: max_themes * keywords_per_theme]
    themes: List[ThemeResult] = []

    for i in range(0, len(top_terms), keywords_per_theme):
        kw = top_terms[i : i + keywords_per_theme]
        if not kw:
            continue
        theme_name = ", ".join(kw[: min(3, len(kw))])

        # pick sample titles that contain at least one keyword term
        samples: List[str] = []
        for rt in raw_titles:
            rt_clean = _clean_text(rt)
            if any(k in rt_clean for k in kw):
                samples.append(rt)
            if len(samples) >= samples_per_theme:
                break

        themes.append(ThemeResult(theme=theme_name, keywords=kw, sample_titles=samples))

    # Limit themes
    themes = themes[:max_themes]

    return {
        "themes": [
            {
                "theme": t.theme,
                "keywords": t.keywords,
                "sample_titles": t.sample_titles,
            }
            for t in themes
        ],
        "note": "Themes are derived from headline text using TF-IDF (explainable, keyless).",
    }
