# System Architecture

The RAG architecture can be visualized in three phases:

1. **Ingestion Phase**
   - Documents are read and chunked.
   - Each chunk is converted into a vector embedding using a sentence transformer.
   - These embeddings are stored in a vector database such as Chroma or FAISS.

2. **Retrieval Phase**
   - The user query is embedded using the same model.
   - The system performs similarity search to retrieve top relevant chunks.

3. **Generation Phase**
   - The retrieved chunks are passed to an LLM such as Llama 3.
   - The LLM generates a response grounded in the retrieved context.
