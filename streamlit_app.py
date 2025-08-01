# streamlit_preprocess_app.py
"""
Instagram Caption Pre-processing App (Streamlit)
------------------------------------------------
Upload an **`ig_posts_raw_mini.csv`**-style file and the app will:
1. Rename **`shortcode` → `ID`** and **`caption` → `Context`**.
2. Split every caption into sentences using a custom Instagram-aware tokenizer (handles emojis, ellipses, etc.).
3. Output four columns exactly matching the sample transformed file: `ID`, `Context`, `Statement`, `Sentence ID`.
4. Let you preview the transformed dataset and download it as CSV.
"""

import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------  UI CONFIG  ---------- #
st.set_page_config(page_title="Instagram Caption Pre-processor", layout="wide")

st.title("📸 Instagram Caption Pre-processing App")

st.markdown(
    """
    **Step 1 — Upload** an *ig_posts_raw_mini.csv*-style file (must contain **`shortcode`** and **`caption`** columns).  
    **Step 2 — Process** to break each caption into individual sentences.  
    **Step 3 — Download** the transformed CSV that mirrors *ig_posts_transformed_mini.csv*.
    """,
)

# ----------  TOKENIZER  ---------- #

# Regex capturing typical sentence-ending punctuation, including Instagram ellipses (…)
_SENTENCE_END_RE = re.compile(r"([.!?…])+")

def instagram_sentence_tokenize(text: str) -> list[str]:
    """Split a caption into sentences while respecting emojis/ellipses quirks."""
    text = re.sub(r"\s+", " ", str(text).strip())  # normalize whitespace
    if not text:
        return []

    parts = _SENTENCE_END_RE.split(text)
    sentences: list[str] = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        if i + 1 < len(parts):
            sentence += parts[i + 1]  # re-attach delimiter(s)
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)
    return sentences

# ----------  CORE TRANSFORMATION  ---------- #

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw IG post data into the sentence-level format."""

    required_cols = {"shortcode", "caption"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(required_cols - set(df.columns))
        raise ValueError(f"Input CSV missing required column(s): {missing}")

    # Rename to desired schema
    df =
