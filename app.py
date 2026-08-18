
import os
import re
import json
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "pbls_faiss.index")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "pbls_embeddings_final.json")
FINAL_FILE = os.path.join(BASE_DIR, "pbls_final_ready.json")
VISUAL_DIR = os.path.join(BASE_DIR, "pbls_visuals")


# =========================
# Load PBLS data
# =========================
@st.cache_resource
def load_resources():

    index = faiss.read_index(INDEX_FILE)

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)

    chunks = embeddings_data["chunks"]

    with open(FINAL_FILE, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    qr_mappings = final_data.get("qr_mappings", [])

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    client = genai.Client(api_key=api_key)

    return index, chunks, qr_mappings, model, client


index, chunks, QR_MAPPINGS, model, client = load_resources()


# =========================
# Retrieval
# =========================
def retrieve_pbls(query, top_k=3):

    q = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        q,
        min(top_k, index.ntotal)
    )

    results = []

    for score, i in zip(scores[0], indices[0]):

        if i >= 0:
            results.append({
                "score": float(score),
                "chunk": chunks[i]
            })

    return results


# =========================
# RAG context
# =========================
def build_rag_context(query, top_k=3):

    results = retrieve_pbls(
        query,
        top_k=top_k
    )

    context_parts = []

    for r in results:

        c = r["chunk"]

        context_parts.append(
            f"[PBLS SOURCE - Page {c['page']}]\n"
            f"{c['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


# =========================
# PBLS chatbot
# =========================
def pbls_chat(query, top_k=3):

    context = build_rag_context(
        query,
        top_k=top_k
    )

    prompt = f"""
You are a medical educational chatbot specialized ONLY in Paediatric Basic Life Support (PBLS).

Answer ONLY from the provided PBLS context.

Do not add outside medical information.

If the answer is not found in the context, say exactly:

"Information not found in the provided PBLS source."

Give a clear educational answer and include the relevant page number(s).

PBLS CONTEXT:
{context}

USER QUESTION:
{query}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# =========================
# Find related visuals
# =========================
def get_pages(text):

    pages = sorted(
        set(
            int(x)
            for x in re.findall(
                r"[Pp]age\s+(\d+)",
                text
            )
            if 1 <= int(x) <= 30
        )
    )

    return pages


def get_visuals(pages):

    paths = []

    if not os.path.exists(VISUAL_DIR):
        return paths

    for page in pages:

        for filename in os.listdir(VISUAL_DIR):

            if f"page_{page}_" in filename.lower():

                paths.append(
                    os.path.join(
                        VISUAL_DIR,
                        filename
                    )
                )

    return paths


def get_qrs(pages):

    return [
        qr for qr in QR_MAPPINGS
        if qr.get("page") in pages
    ]


# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="PBLS Educational Chatbot",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 PBLS Educational Chatbot")

st.caption(
    "Educational chatbot based ONLY on the ERC Paediatric Basic Life Support source."
)

question = st.text_area(
    "PBLS Question",
    placeholder="Ask a question about Paediatric Basic Life Support...",
    height=120
)

if st.button("Submit", type="primary"):

    if not question.strip():

        st.warning("Please enter a PBLS question.")

    else:

        with st.spinner("Searching PBLS source..."):

            try:

                answer = pbls_chat(
                    question,
                    top_k=3
                )

                st.markdown("## Answer")
                st.markdown(answer)

                pages = get_pages(answer)

                if pages:

                    st.markdown("## 📚 Relevant Pages")
                    st.write(
                        ", ".join(
                            f"Page {p}"
                            for p in pages
                        )
                    )

                visuals = get_visuals(pages)

                if visuals:

                    st.markdown("## 🖼️ Related PBLS Visuals")

                    for image_path in visuals:

                        st.image(
                            image_path,
                            use_container_width=True
                        )

                qrs = get_qrs(pages)

                if qrs:

                    st.markdown("## 🔗 Related QR References")

                    for qr in qrs:

                        st.info(
                            f"QR {qr['qr']} — "
                            f"{qr['topic'].replace('_', ' ').title()} — "
                            f"Page {qr['page']} — "
                            f"{qr['age']}"
                        )

            except Exception as e:

                st.error(
                    f"Application error: {e}"
                )
