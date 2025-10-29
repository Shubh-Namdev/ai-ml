from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import requests
from config import (
    GENERATOR_MODE, OLLAMA_URL, OLLAMA_MODEL,
    HUGGINGFACE_MODEL, USE_HF_API, HUGGINGFACE_API_TOKEN
)
from query import retrieve_relevant_docs

app = FastAPI(title="Mini RAG API")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

def generate_answer_huggingface(prompt: str) -> str:
    try:
        if USE_HF_API:
            # --- Option A: Using Hugging Face Inference API ---
            url = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"
            print(url)
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}

            response = requests.post(url,headers=headers, json=payload)
            if response.status_code != 200:
                return f"Error from Hugging Face API: {response.text}"
            data = response.json()
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            return str(data)
        else:
            # --- Option B: Using Local Model via Transformers ---
            generator = pipeline("text-generation", model=HUGGINGFACE_MODEL)
            result = generator(prompt, max_new_tokens=200, do_sample=True)
            return result[0]["generated_text"].strip()
    except Exception as e:
        return f"Error using Hugging Face model: {e}"

def generate_answer_ollama(prompt: str) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        return f"Error from Ollama: {response.text}"
    return response.json().get("response", "").strip()

@app.post("/ask")
def ask_question(req: QueryRequest):
    print("generating answer......")
    docs, sources = retrieve_relevant_docs(req.question, top_k=req.top_k)

    context = "\n\n".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {req.question}\n\nAnswer based only on the context above."

    if GENERATOR_MODE == "OLLAMA":
        answer = generate_answer_ollama(prompt)
    elif GENERATOR_MODE == "HUGGINGFACE":
        answer = generate_answer_huggingface(prompt)
    else:
        answer = "(LLM not configured - showing context only)\n" + context[:1000]

    return {"question": req.question, "answer": answer, "sources": sources}
