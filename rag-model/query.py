import sys
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, EMBED_MODEL_NAME

def load_collection():
    client = PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_collection("rag_collection")

def retrieve_relevant_docs(query_text, top_k=3):
    # print(f"\n[Query] Searching top {top_k} matches for: {query_text!r}")

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    query_embedding = embedder.encode([query_text], convert_to_numpy=True).tolist()

    collection = load_collection()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    # print("\n[Top matches]:")
    # for i, (doc, src) in enumerate(zip(docs, sources), 1):
    #     print(f"{i}. Source: {src}\n   {doc[:200]}...\n")

    return docs, sources

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py <your question>")
        sys.exit(1)

    query_text = " ".join(sys.argv[1:])
    retrieve_relevant_docs(query_text)
