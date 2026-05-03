"""Ingest PDFs from data/ into a Chroma vector database."""
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
import voyageai

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_PARAGRAPH_CHUNK = 2000
EMBED_MODEL = "voyage-3"
BATCH_SIZE = 64

# Invisible marker used during ingestion to track page numbers across
# the full-document text used for paragraph chunking. Null bytes don't
# appear in real PDF text so they won't collide with content.
PAGE_MARKER = "\x00PAGE:{}\x00"

# Filename -> human-readable title
DOCUMENT_TITLES = {
    # Fill in as you identify each PDF, e.g.:
    # "Bekendtgørelse af lov om miljøbeskyttelse LBK1093.pdf":
    #     "Miljøbeskyttelsesloven (LBK 1093 af 2024)",
}

# Files that should use structure-aware chunking (Danish legal docs)
LEGAL_DOCS = {
    "Bekendtgørelse af lov om miljøbeskyttelse LBK1093.pdf",
    "Bekendtgørelse af lov om vandforsyning m.v..pdf",
    "Bekendtgørelse om bygningsreglement 2018 (BR18).pdf",
    "Bekendtgørelse om jordvarmeanlæg.pdf",
    "Bekendtgørelse om miljøregulering af dyrehold og om opbevaring og anvendelse af gødning.pdf",
    "Bekendtgørelse om vandkvalitet og tilsyn med vandforsyningsanlæg.pdf",
}


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) tuples for a PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def extract_with_page_markers(pdf_path: Path) -> str:
    """Extract all pages joined together, with invisible page markers
    so paragraph chunks can later report which page they came from."""
    reader = PdfReader(pdf_path)
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        parts.append(PAGE_MARKER.format(i) + text)
    return "\n".join(parts)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of roughly `size` characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def chunk_by_paragraph_with_pages(text: str) -> list[tuple[str, int]]:
    """Split Danish legal text on § boundaries.
    Returns list of (chunk_text_without_markers, first_page_number) tuples."""
    parts = re.split(r"(?=^§\s*\d+[a-zæøå]?\.)", text, flags=re.MULTILINE)
    marker_pattern = re.compile(r"\x00PAGE:(\d+)\x00")
    results = []
    current_page = 1  # default if no marker seen yet

    for p in parts:
        if not p.strip():
            continue

        # The first page marker in this chunk tells us where this § starts
        markers_in_chunk = list(marker_pattern.finditer(p))
        if markers_in_chunk:
            current_page = int(markers_in_chunk[0].group(1))

        # Strip all page markers from the chunk text before storing
        clean = marker_pattern.sub("", p).strip()
        if not clean:
            continue

        # Long-paragraph fallback: split very long §§ into smaller chunks
        if len(clean) > MAX_PARAGRAPH_CHUNK:
            for sub in chunk_text(clean, CHUNK_SIZE, CHUNK_OVERLAP):
                results.append((sub, current_page))
        else:
            results.append((clean, current_page))

    return results


def main():
    voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)

    # Reset the collection so re-running this script gives a clean state
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(COLLECTION_NAME)

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}/")
        return

    print(f"Found {len(pdfs)} PDFs. Processing...")

    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_id = 0

    for pdf_path in pdfs:
        title = DOCUMENT_TITLES.get(pdf_path.name, pdf_path.stem)
        is_legal = pdf_path.name in LEGAL_DOCS

        if is_legal:
            print(f"  Reading {pdf_path.name} (paragraph-aware)...")
            full_text = extract_with_page_markers(pdf_path)
            chunks_with_pages = chunk_by_paragraph_with_pages(full_text)
            for chunk, page_num in chunks_with_pages:
                m = re.match(r"§\s*(\d+[a-zæøå]?)", chunk)
                paragraph = m.group(1) if m else None

                enriched_chunk = (
                    f"Document: {title}\n"
                    f"Page: {page_num}\n"
                    + (f"Paragraph: § {paragraph}\n" if paragraph else "")
                    + f"---\n{chunk}"
                )
                all_chunks.append(enriched_chunk)
                all_metadatas.append({
                    "source": pdf_path.name,
                    "title": title,
                    "page": page_num,
                    "paragraph": paragraph or "",
                })
                all_ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
        else:
            print(f"  Reading {pdf_path.name} (character-based)...")
            pages = extract_pages(pdf_path)
            for page_num, page_text in pages:
                for chunk in chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
                    enriched_chunk = (
                        f"Document: {title}\n"
                        f"Page: {page_num}\n"
                        f"---\n{chunk}"
                    )
                    all_chunks.append(enriched_chunk)
                    all_metadatas.append({
                        "source": pdf_path.name,
                        "title": title,
                        "page": page_num,
                        "paragraph": "",
                    })
                    all_ids.append(f"chunk_{chunk_id}")
                    chunk_id += 1

    print(f"Total chunks: {len(all_chunks)}")
    print("Embedding (this may take a minute)...")

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        result = voyage.embed(batch, model=EMBED_MODEL, input_type="document")
        collection.add(
            ids=all_ids[i:i + BATCH_SIZE],
            documents=batch,
            embeddings=result.embeddings,
            metadatas=all_metadatas[i:i + BATCH_SIZE],
        )
        print(f"  Embedded {min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)}")

    print(f"\nDone. Indexed {len(all_chunks)} chunks from {len(pdfs)} PDFs.")
    print(f"Database stored in: {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
