# streamlit_preprocess_app.py
"""
Instagram Caption Pre‑processing App (Streamlit)
------------------------------------------------
Upload an **`ig_posts_raw_mini.csv`**‑style file and the app will:
1. Rename **`shortcode` → `ID`** and **`caption` → `Context`**.
2. Split every caption into sentences using a custom Instagram‑aware tokenizer (handles emojis, ellipses, etc.).
3. Output four columns that mirror *ig_posts_transformed_mini.csv*: `ID`, `Context`, `Statement`, `Sentence ID`.
4. Preview the transformed data and download it as CSV.

**Tip:** The file‑upload control now lives in the left‑hand sidebar under *“📂 Upload CSV”*.
"""

import re
from typing import List, Dict

import pandas as pd
import streamlit as st

# ----------  UI CONFIG  ---------- #
st.set_page_config(
    page_title="Instagram Caption Pre‑processor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📸 Instagram Caption Pre‑processing App")

st.markdown(
    """
    **Step 1 — Upload** an **ig_posts_raw_mini.csv**‑style file (must contain `shortcode` and `caption` columns) using the **📂 Upload CSV** control in the sidebar.  
    **Step 2 — Process** to break each caption into individual sentences.  
    **Step 3 — Download** the transformed CSV.
    """,
)

# ----------  TOKENIZER  ---------- #

_SENTENCE_END_RE = re.compile(r"([.!?…])+")  # handles ellipses & repeated punctuation

def instagram_sentence_tokenize(text: str) -> List[str]:
    """Break a caption into sentences, respecting emojis/Instagram quirks."""
    text = re.sub(r"\s+", " ", str(text).strip())  # normalize whitespace
    if not text:
        return []

    parts = _SENTENCE_END_RE.split(text)
    sentences: List[str] = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        if i + 1 < len(parts):
            sentence += parts[i + 1]  # re‑attach delimiter(s)
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)
    return sentences

# ----------  CORE TRANSFORMATION  ---------- #

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw IG post data into the sentence‑level format."""

    required_cols = {"shortcode", "caption"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(required_cols - set(df.columns))
        raise ValueError(f"Input CSV missing required column(s): {missing}")

    df = df.rename(columns={"shortcode": "ID", "caption": "Context"})

    records: List[Dict] = []
    for _, row in df.iterrows():
        post_id = row["ID"]
        context = str(row["Context"])
        for sent_idx, sentence in enumerate(instagram_sentence_tokenize(context), start=1):
            records.append({
                "ID": post_id,
                "Context": context,
                "Statement": sentence,
                "Sentence ID": sent_idx,
            })

    return pd.DataFrame(records)

# ----------  SIDEBAR  ---------- #
st.sidebar.header("⚙️ Options")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type="csv", help="Must include 'shortcode' and 'caption' columns")
max_rows = st.sidebar.slider("Max rows to display", min_value=10, max_value=500, value=100, step=10)

# ----------  MAIN WORKFLOW  ---------- #

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        processed_df = preprocess(raw_df)

        st.success("✅ Processing complete! Preview below.")
        st.dataframe(processed_df.head(max_rows), use_container_width=True, height=600)

        st.download_button(
            label="💾 Download transformed CSV",
            data=processed_df.to_csv(index=False).encode("utf‑8"),
            file_name="ig_posts_transformed.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("➡️ Use the **📂 Upload CSV** control in the left sidebar to get started.")
