import sys
import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, EMBED_MODEL_NAME

# ========== Helper Functions ==========

def read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks

# ========== Main Ingest Logic ==========

def ingest_folder(folder: Path):
    print(f"\n[Ingest] Using embedding model: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    client = PersistentClient(path=CHROMA_PERSIST_DIR)

    # create or get collection
    if 'rag_collection' in [c.name for c in client.list_collections()]:
        collection = client.get_collection('rag_collection')
    else:
        collection = client.create_collection('rag_collection')

    files = list(folder.glob('**/*'))
    text_files = [f for f in files if f.suffix.lower() in ['.txt', '.md']]

    if not text_files:
        print(f"No text files found in {folder}")
        return

    for file_path in text_files:
        print(f"\nProcessing: {file_path.name}")
        text = read_text_file(file_path)
        chunks = chunk_text(text)
        if not chunks:
            continue
        ids = [f"{file_path.stem}_{uuid.uuid4().hex[:6]}_{i}" for i in range(len(chunks))]
        embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True).tolist()
        metadatas = [{"source": file_path.name, "chunk_index": i} for i in range(len(chunks))]

        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        print(f"Added {len(chunks)} chunks from {file_path.name}")

    # client.persist()
    print("\n✅ Ingestion completed.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <folder_with_txt_docs>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    ingest_folder(folder)
