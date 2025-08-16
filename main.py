from fastapi import FastAPI
from pydantic import BaseModel
import re
from sentence_transformers import SentenceTransformer, util


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

def get_similarity(text1: str, text2: str) -> float:
    model = get_model()
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return round((score + 1) / 2, 3)

def get_similarity(text1: str, text2: str) -> float:
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return round((score + 1) / 2, 3)  # normalize to range 0–1

# ---------- FastAPI app ----------
app = FastAPI(title="Text Similarity API")

class SimilarityRequest(BaseModel):
    text1: str
    text2: str

@app.get("/")
def root():
    return {"message": "Welcome to the Text Similarity API! Use POST /similarity to get scores."}

@app.post("/similarity")
def similarity_endpoint(req: SimilarityRequest):
    clean1 = req.text1
    clean2 = req.text2
    score = get_similarity(clean1, clean2)
    return {"similarity score": score}

