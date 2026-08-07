import json

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

st.set_page_config(page_title="Movie Detail Extractor", page_icon="🎬")
st.title("🎬 Movie Detail Extractor")
st.caption("Paste a paragraph about a movie and extract structured details from it.")


@st.cache_resource
def get_model():
    return init_chat_model("mistralai:mistral-small-2603")


model = get_model()

PROMPT_TEMPLATE = """From the given paragraph Extract the following details and return the response in JSON format.
Movie
Title
Genre
Release Year
Director
Lead Cast
Music Composer
Core Plot / Premise
Key Settings & Themes
Audience / Critical Reception
Review Ratings

Here is the paragraph:
{paragraph}
"""

paragraph = st.text_area(
    "Enter the paragraph:",
    height=200,
    placeholder="Paste a paragraph describing a movie here...",
)

extract_clicked = st.button("Extract Details", type="primary", disabled=not paragraph.strip())

if extract_clicked:
    prompt = PROMPT_TEMPLATE.format(paragraph=paragraph)

    with st.spinner("Extracting details..."):
        response = model.invoke(prompt)

    raw_content = response.content

    def clean_json_text(text: str) -> str:
        """Strip code fences, stray labels, and any text outside the outermost {}."""
        cleaned = text.strip()

        # Remove ```json ... ``` or ``` ... ``` fences
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        # Keep only the outermost { ... } block, dropping any preamble/explanation text
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        return cleaned.strip()

    cleaned = clean_json_text(raw_content)

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    st.subheader("Extracted Details")

    if parsed is not None:
        st.json(parsed)

        with st.expander("View as table"):
            st.table(
                {
                    "Field": list(parsed.keys()),
                    "Value": [str(v) for v in parsed.values()],
                }
            )

        download_data = json.dumps(parsed, indent=2)
        download_name = "movie_details.json"
        mime_type = "application/json"
    else:
        st.warning("Couldn't parse the response as JSON. Showing cleaned output instead.")
        st.code(cleaned, language="json")

        download_data = cleaned
        download_name = "movie_details.txt"
        mime_type = "text/plain"

    st.download_button(
        label="⬇️ Download response",
        data=download_data,
        file_name=download_name,
        mime=mime_type,
    )

    with st.expander("Raw model response"):
        st.code(raw_content, language="json")