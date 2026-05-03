"""Ask questions against the indexed documents."""
import os
from dotenv import load_dotenv
import chromadb
import voyageai
from anthropic import Anthropic

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "voyage-3"
LLM_MODEL = "claude-sonnet-4-6"
TOP_K = 5


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context from documents.

Rules:
- Answer ONLY using information from the context below.
- If the context doesn't contain the answer, say so clearly. Do not make things up.
- Cite your sources inline using the format [source: filename, page X].
- Be concise but complete."""


def retrieve(question: str, voyage, collection, k: int = TOP_K):
    """Embed the question and retrieve top-k chunks."""
    result = voyage.embed([question], model=EMBED_MODEL, input_type="query")
    query_embedding = result.embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    return list(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ))


def format_context(chunks) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    parts = []
    for i, (text, meta, _) in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}: {meta['source']}, page {meta['page']}]\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def answer(question: str, voyage, collection, anthropic):
    chunks = retrieve(question, voyage, collection)
    context = format_context(chunks)

    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    response = anthropic.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text, chunks


def main():
    voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection(COLLECTION_NAME)
    anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"Loaded {collection.count()} chunks. Ready for questions.\n")
    print("Type your question (or 'quit' to exit):\n")

    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        try:
            response, chunks = answer(question, voyage, collection, anthropic)
            print(f"\n{response}\n")
            print("Sources retrieved:")
            for i, (_, meta, dist) in enumerate(chunks, start=1):
                print(f"  [{i}] {meta['source']}, page {meta['page']} (distance: {dist:.3f})")
            print()
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
