"""Streamlit UI for the Ask Your Documents RAG bot."""
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import chromadb
import voyageai
from anthropic import Anthropic

# Make src importable
sys.path.insert(0, str(Path(__file__).parent / "src"))
from query import format_context, SYSTEM_PROMPT, LLM_MODEL, COLLECTION_NAME, CHROMA_DIR
from hybrid import build_bm25_index, hybrid_retrieve

load_dotenv()

# ---------- Page config ----------
st.set_page_config(
    page_title="Ask Your Documents",
    page_icon="📄",
    layout="wide",
)


# ---------- Cached resources ----------
@st.cache_resource
def get_clients():
    """Initialize and cache API clients, Chroma collection, and BM25 index."""
    voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = chroma.get_collection(COLLECTION_NAME)
        bm25, bm25_docs, bm25_metadatas, bm25_ids = build_bm25_index(collection)
    except Exception:
        collection = None
        bm25 = bm25_docs = bm25_metadatas = bm25_ids = None
    anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return voyage, collection, anthropic, bm25, bm25_docs, bm25_metadatas, bm25_ids


def stream_answer(
    question: str,
    voyage,
    collection,
    anthropic,
    top_k: int,
    alpha: float,
    bm25,
    bm25_docs,
    bm25_metadatas,
    bm25_ids,
):
    """Retrieve chunks via hybrid search and stream the LLM response."""
    chunks = hybrid_retrieve(
        question, voyage, collection,
        bm25, bm25_docs, bm25_metadatas, bm25_ids,
        k=top_k, alpha=alpha,
    )
    context = format_context(chunks)
    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    with anthropic.messages.stream(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            yield text, chunks


# ---------- Initialize ----------
voyage, collection, anthropic, bm25, bm25_docs, bm25_metadatas, bm25_ids = get_clients()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("📚 Index status")
    if collection is None:
        st.error("No index found. Run `uv run src/ingest.py` first.")
    else:
        st.metric("Chunks indexed", collection.count())

        all_metadata = collection.get()["metadatas"]
        unique_sources = sorted({m["source"] for m in all_metadata})
        st.metric("Documents", len(unique_sources))

        with st.expander("View documents"):
            for src in unique_sources:
                st.text(f"• {src}")

    st.divider()
    st.header("⚙️ Retrieval settings")

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1, max_value=15, value=5,
    )

    alpha = st.slider(
        "Vector vs keyword balance (α)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
        help=(
            "1.0 = pure vector search (semantic similarity). "
            "0.0 = pure BM25 (keyword matching). "
            "0.5 = equal mix. Lower α helps for queries with specific symbols like § or law numbers."
        ),
    )

    st.divider()
    if st.button("🔄 Rebuild index", help="Re-runs ingestion. Takes 1-3 minutes."):
        with st.spinner("Rebuilding index..."):
            import subprocess
            result = subprocess.run(
                ["uv", "run", "src/ingest.py"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                st.success("Index rebuilt. Refresh the page.")
                st.cache_resource.clear()
            else:
                st.error(f"Failed:\n{result.stderr}")


# ---------- Main area ----------
st.title("📄 Ask Your Documents")
st.caption("Query your document corpus using retrieval-augmented generation.")

if collection is None or collection.count() == 0:
    st.warning("Index is empty. Add PDFs to `data/` and rebuild the index.")
    st.stop()

# Question input
question = st.text_input(
    "Your question",
    placeholder="e.g., Hvad siger § 23 i miljøbeskyttelsesloven?",
)

# Example questions
with st.expander("💡 Example questions"):
    examples = [
        "What are the EU requirements for groundwater quality monitoring?",
        "How does the Nitrates Directive define vulnerable zones?",
        "Hvad siger § 23 i miljøbeskyttelsesloven?",
        "How is the EU Water Framework Directive implemented in Danish law?",
    ]
    for ex in examples:
        st.markdown(f"- {ex}")


# ---------- Run query ----------
if question:
    col_answer, col_sources = st.columns([3, 2])

    with col_answer:
        st.subheader("Answer")
        answer_placeholder = st.empty()

        full_answer = ""
        chunks = None
        try:
            for text_delta, retrieved in stream_answer(
                question, voyage, collection, anthropic, top_k, alpha,
                bm25, bm25_docs, bm25_metadatas, bm25_ids,
            ):
                full_answer += text_delta
                chunks = retrieved
                answer_placeholder.markdown(full_answer + "▌")
            answer_placeholder.markdown(full_answer)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    with col_sources:
        st.subheader("Sources")
        if chunks:
            for i, (text, meta, distance) in enumerate(chunks, start=1):
                # Lower distance = more relevant. Color-code roughly.
                if distance < 0.4:
                    badge = "🟢"
                elif distance < 0.7:
                    badge = "🟡"
                else:
                    badge = "🔴"

                with st.expander(
                    f"{badge} [{i}] {meta['source']} — page {meta['page']} "
                    f"(distance: {distance:.3f})"
                ):
                    st.text(text)
