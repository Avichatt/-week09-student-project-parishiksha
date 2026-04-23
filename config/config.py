# PariShiksha — Central Configuration

# All project-wide constants, paths, model names, and hyperparameters are present here.



import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TOKENIZER_OUTPUT_DIR = OUTPUTS_DIR / "tokenizer_comparison"
CHUNKING_OUTPUT_DIR = OUTPUTS_DIR / "chunking_analysis"
RETRIEVAL_OUTPUT_DIR = OUTPUTS_DIR / "retrieval_results"
EVAL_OUTPUT_DIR = OUTPUTS_DIR / "evaluation_reports"

# Ensure all directories exist
for dir_path in [
    RAW_DATA_DIR, EXTRACTED_DATA_DIR, PROCESSED_DATA_DIR,
    EVALUATION_DATA_DIR, TOKENIZER_OUTPUT_DIR, CHUNKING_OUTPUT_DIR,
    RETRIEVAL_OUTPUT_DIR, EVAL_OUTPUT_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)


# API KEYS

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# NCERT CHAPTER CONFIGURATION

# Chapters we are processing (Class 9 Science)
TARGET_CHAPTERS = {
    "chapter_4": {
        "title": "Describing Motion Around Us",
        "class": 9,
        "subject": "Science",
        "pdf_filename": "iesc104.pdf",
        "page_range": None,
    },
}

# NCERT PDF download URLs (official NCERT website)
NCERT_BASE_URL = "https://ncert.nic.in/textbook/pdf"


# TOKENIZER MODELS (for comparison in Stage 1)

TOKENIZER_MODELS = {
    "bert_base": "bert-base-uncased",
    "bert_bio": "dmis-lab/biobert-base-cased-v1.2",
    "t5_base": "t5-base",
    "gpt2": "gpt2",
}

# Scientific terms to test tokenizer behavior on
SCIENCE_TERMS = [
    "photosynthesis",
    "mitochondria",
    "endoplasmic reticulum",
    "specific heat capacity",
    "electromagnetic spectrum",
    "nucleotide",
    "chlorophyll",
    "ribonucleic acid",
    "deoxyribonucleic acid",
    "prokaryotic",
    "eukaryotic",
    "Golgi apparatus",
    "osmosis",
    "plasmolysis",
    "chromatin",
    "cytoplasm",
]


# CHUNKING PARAMETERS

CHUNKING_CONFIG = {
    "strategies": ["fixed_token", "sentence_based", "semantic_paragraph"],
    "fixed_token_sizes": [128, 256, 512],   # tokens per chunk
    "overlap_ratio": 0.15,                   # 15% overlap between chunks
    "min_chunk_tokens": 50,                  # discard chunks smaller than this
    "sentence_group_size": 5,                # sentences per chunk in sentence strategy
}


# EMBEDDING & RETRIEVAL PARAMETERS

EMBEDDING_CONFIG = {
    # Dense embedding model (SBERT)
    "dense_model": "all-MiniLM-L6-v2",
    # Number of top chunks to retrieve
    "top_k": 5,
    # Similarity metric
    "similarity_metric": "cosine",
    # TF-IDF parameters (sparse baseline)
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": (1, 2),
}


# GENERATION PARAMETERS

GENERATION_CONFIG = {
    # Gemini (decoder-only)
    "gemini_model": "models/gemini-flash-latest",
    "gemini_temperature": 0.3,         # Low temperature for factual grounding
    "gemini_max_tokens": 512,
    # T5 (encoder-decoder, for Stretch)
    "t5_model": "google/flan-t5-base",
    "t5_max_length": 256,
    "t5_num_beams": 4,
    # Grounding prompt template
    "system_prompt": (
        "You are a study assistant for Class 9 and 10 NCERT Science students. "
        "Answer the question using ONLY the provided textbook context. "
        "If the context does not contain enough information to answer, say: "
        "'I don't have enough information from the textbook to answer this.' "
        "Do not add information beyond what is in the context. "
        "Use simple language appropriate for a Class 9-10 student."
    ),
}


#  EVALUATION PARAMETERS

EVALUATION_CONFIG = {
    # Minimum number of evaluation questions
    "min_eval_questions": 20,
    # Question types to include
    "question_types": [
        "factual",           # Direct fact recall
        "conceptual",        # Understanding of concepts
        "application",       # Apply concept to new scenario
        "unanswerable",      # Not in the textbook (test hallucination)
        "hindi_codeswitched", # Hindi-English mixed queries
    ],
    # Metrics
    "metrics": ["faithfulness", "answer_relevance", "retrieval_precision"],
    # BERTScore model
    "bertscore_model": "microsoft/deberta-xlarge-mnli",
}


# LOGGING

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}"
