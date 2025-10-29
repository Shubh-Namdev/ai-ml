from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables (optional but recommended)
load_dotenv()

# Paths
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")

# Model Names
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM Options: 'OLLAMA', 'OPENAI', 'DUMMY'
# GENERATOR_MODE = "OLLAMA"
GENERATOR_MODE = "HUGGINGFACE"

# --- Hugging Face Settings ---
# For local inference (via transformers)
# HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct"  # try "google/flan-t5-base" if you want lightweight

HUGGINGFACE_MODEL = "gpt2"
USE_HF_API = False  # True = use Hugging Face Inference API
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Ollama settings
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"