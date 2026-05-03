"""Check whether specific § references exist in the indexed corpus."""
import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"


def main():
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection(COLLECTION_NAME)
    data = collection.get()
    docs = data["documents"]
    metas = data["metadatas"]

    print(f"Total chunks in index: {len(docs)}\n")

    # Find chunks that contain "§ 23." (with period) on early pages
    # — i.e. likely the actual definition, not a cross-reference
    print("=" * 70)
    print("Chunks with '§ 23.' (period) on pages < 50 — likely definitions:")
    print("=" * 70)

    definition_matches = [
        (i, d, m)
        for i, (d, m) in enumerate(zip(docs, metas))
        if "§ 23." in d and m["page"] < 50
    ]
    print(f"Found {len(definition_matches)} candidates\n")

    for i, d, m in definition_matches:
        print(f"\n--- chunk #{i}, {m['source']} p.{m['page']} ---")
        print(d[:600])
        print("...")

    # Also show all chunks with "§ 23" (cross-references and definitions combined)
    # to compare ranking — the definition chunk should be findable
    print("\n\n" + "=" * 70)
    print("All chunks containing '§ 23' (any form), grouped by source:")
    print("=" * 70)
    from collections import defaultdict
    by_source = defaultdict(list)
    for i, (d, m) in enumerate(zip(docs, metas)):
        if "§ 23" in d:
            by_source[m["source"]].append((i, m["page"]))
    for src, hits in sorted(by_source.items()):
        pages = sorted({p for _, p in hits})
        print(f"\n  {src}: {len(hits)} chunks on pages {pages}")


if __name__ == "__main__":
    main()
