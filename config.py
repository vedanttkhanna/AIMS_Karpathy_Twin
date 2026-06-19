import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")



# --- Model ---
GEMINI_MODEL = "gemini-2.5-flash"

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- RAG Settings ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_SEMANTIC = 15
TOP_K_BM25 = 15
TOP_K_FINAL = 6

# --- Memory ---
SHORT_TERM_WINDOW = 10
MEMORY_DB_PATH = "memory_store/memory.db"

# --- Paths ---
RAW_DATA_DIR = "data/raw"
CHROMA_DIR = "data/chroma"
KNOWLEDGE_BASE_DIR = "knowledge_base"

# --- Sources to scrape ---


GITHUB_READMES = [
    # existing...
    "https://raw.githubusercontent.com/karpathy/nanoGPT/master/README.md",
    "https://raw.githubusercontent.com/karpathy/micrograd/master/README.md",
    "https://raw.githubusercontent.com/karpathy/makemore/master/README.md",
    "https://raw.githubusercontent.com/karpathy/llm.c/master/README.md",
    "https://raw.githubusercontent.com/karpathy/minbpe/master/README.md",
    # new
    "https://raw.githubusercontent.com/karpathy/nn-zero-to-hero/master/README.md",
    "https://raw.githubusercontent.com/karpathy/randomfun/master/README.md",
]

YOUTUBE_VIDEO_IDS = [
    "VMj-3S1tku0",
    "PaCmpygFfXo",
    "TCH_1BHY58I",
    "P6sfmUTpUmc",
    "q8SA3rM6ckI",
    "t3YJ5hKiMQ0",
    "kCc8FmEb1nY",
    "l8pRSuU81PU",
    "zduSFxRajkE",
]
KARPATHY_BLOG_URLS = [
    # existing ones...
    "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
    "https://karpathy.github.io/2019/04/25/recipe/",
    "https://karpathy.github.io/2022/03/14/lecun1989/",
    "https://karpathy.github.io/neuralnets/",
    "https://karpathy.github.io/2015/03/30/breaking-convnets/",
    # new
    "https://karpathy.github.io/2022/03/14/lecun1989/",
    "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
]
