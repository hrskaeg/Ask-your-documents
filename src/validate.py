"""Validate that all PDFs were processed and indexed correctly."""
from pathlib import Path
from collections import defaultdict
import chromadb
from pypdf import PdfReader

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"


def main():
    print("=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    # --- Stage 1: PDFs on disk ---
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    print(f"\n[1] PDFs in {DATA_DIR}/: {len(pdfs)}")
    for p in pdfs:
        size_kb = p.stat().st_size / 1024
        print(f"    • {p.name} ({size_kb:.1f} KB)")

    # --- Stage 2: PDF extraction (pypdf) ---
    print(f"\n[2] PDF extraction (pypdf):")
    extraction_issues = []
    for p in pdfs:
        try:
            reader = PdfReader(p)
            total_pages = len(reader.pages)
            pages_with_text = 0
            total_chars = 0
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages_with_text += 1
                    total_chars += len(text)

            status = "OK"
            if pages_with_text == 0:
                status = "⚠ NO TEXT (likely scanned PDF — needs OCR)"
                extraction_issues.append(p.name)
            elif pages_with_text < total_pages * 0.5:
                status = f"⚠ Only {pages_with_text}/{total_pages} pages have text"
                extraction_issues.append(p.name)

            print(f"    • {p.name}: {pages_with_text}/{total_pages} pages, "
                  f"{total_chars:,} chars — {status}")
        except Exception as e:
            print(f"    • {p.name}: ❌ ERROR - {e}")
            extraction_issues.append(p.name)

    # --- Stage 3: Chroma index check ---
    print(f"\n[3] Chroma index:")
    try:
        chroma = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma.get_collection(COLLECTION_NAME)
        total_chunks = collection.count()
        print(f"    Total chunks indexed: {total_chunks}")
    except Exception as e:
        print(f"    ❌ Could not load collection: {e}")
        return

    # --- Stage 4: Per-source chunk counts ---
    print(f"\n[4] Chunks per document:")
    data = collection.get()
    by_source = defaultdict(int)
    by_source_pages = defaultdict(set)
    for meta in data["metadatas"]:
        by_source[meta["source"]] += 1
        by_source_pages[meta["source"]].add(meta["page"])

    pdf_names_on_disk = {p.name for p in pdfs}
    pdf_names_in_index = set(by_source.keys())

    for src in sorted(pdf_names_on_disk | pdf_names_in_index):
        if src not in pdf_names_in_index:
            print(f"    ❌ {src}: NOT INDEXED")
        elif src not in pdf_names_on_disk:
            print(f"    ⚠ {src}: in index but PDF missing on disk")
        else:
            n_chunks = by_source[src]
            n_pages = len(by_source_pages[src])
            print(f"    • {src}: {n_chunks} chunks across {n_pages} pages")

    # --- Stage 5: Embedding sanity check ---
    print(f"\n[5] Embedding sanity check:")
    sample = collection.get(limit=5, include=["embeddings", "documents"])
    if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
        dims = len(sample["embeddings"][0])
        print(f"    Embedding dimensions: {dims}")
        zero_count = sum(
            1 for emb in sample["embeddings"]
            if all(v == 0 for v in emb)
        )
        if zero_count > 0:
            print(f"    ⚠ Found {zero_count}/5 zero-vector embeddings (broken)")
        else:
            print(f"    ✓ All sampled embeddings look non-zero")
    else:
        print(f"    ❌ No embeddings found in sample")

    # --- Stage 6: Content spot-check ---
    print(f"\n[6] Content spot-check (Danish chars + symbols):")
    docs = data["documents"]
    checks = {
        "§": sum(1 for d in docs if "§" in d),
        "æ": sum(1 for d in docs if "æ" in d),
        "ø": sum(1 for d in docs if "ø" in d),
        "å": sum(1 for d in docs if "å" in d),
    }
    for char, count in checks.items():
        pct = (count / len(docs) * 100) if docs else 0
        print(f"    '{char}' appears in {count}/{len(docs)} chunks ({pct:.1f}%)")

    # --- Summary ---
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    missing = pdf_names_on_disk - pdf_names_in_index
    print(f"PDFs on disk:        {len(pdf_names_on_disk)}")
    print(f"PDFs in index:       {len(pdf_names_in_index)}")
    print(f"Missing from index:  {len(missing)}")
    print(f"Extraction issues:   {len(extraction_issues)}")
    if missing:
        print(f"\n❌ NOT INDEXED:")
        for m in sorted(missing):
            print(f"    • {m}")
    if extraction_issues:
        print(f"\n⚠ EXTRACTION ISSUES:")
        for e in extraction_issues:
            print(f"    • {e}")
    if not missing and not extraction_issues:
        print(f"\n✓ All PDFs processed successfully")


if __name__ == "__main__":
    main()
