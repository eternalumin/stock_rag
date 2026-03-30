import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk_garbage_key_replace_later")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

LLM_MODEL = "llama-3-70b-8192"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATA_DIR = "data"
KNOWLEDGE_BASE_DIR = "data/knowledge_base"
VECTOR_STORE_DIR = "data/vector_store"

CHROMA_COLLECTION_NAME = "stock_knowledge"

SUPPORTED_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "WMT"]
