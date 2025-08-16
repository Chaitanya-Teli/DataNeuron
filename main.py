from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache


# ---------------------- Model Loading ----------------------

@lru_cache(maxsize=1)  # Cache ensures the model is loaded only once
def get_model():
    """
    Load and return the SentenceTransformer model.
    Using lru_cache prevents re-loading on every request.
    """
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")


def get_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity score between two texts.

    Steps:
    1. Load the model (cached).
    2. Encode both texts into embeddings (vector form).
    3. Calculate cosine similarity between embeddings.
    4. Normalize similarity score from [-1, 1] → [0, 1].
    """
    model = get_model()
    emb1 = model.encode(text1, convert_to_tensor=True)  # vector for text1
    emb2 = model.encode(text2, convert_to_tensor=True)  # vector for text2

    # Cosine similarity between embeddings
    score = util.cos_sim(emb1, emb2).item()

    # Normalize score (cosine similarity is -1 to 1; we scale it to 0–1)
    return round((score + 1) / 2, 3)


# ---------------------- FastAPI App ----------------------

# Create FastAPI instance
app = FastAPI(title="Text Similarity API")


# Define request body schema (using Pydantic)
class SimilarityRequest(BaseModel):
    text1: str
    text2: str


@app.get("/")
def root():
    """
    Root endpoint.
    Provides a welcome message and usage hint.
    """
    return {"message": "Welcome to the Text Similarity API! Use POST /similarity to get scores."}


@app.post("/similarity")
def similarity_endpoint(req: SimilarityRequest):
    
    clean1 = req.text1
    clean2 = req.text2

    # Compute similarity
    score = get_similarity(clean1, clean2)

    return {"similarity score": score}
