from fastapi import FastAPI
from pydantic import BaseModel
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# ---------- NLTK setup ----------
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------- Preprocessing function ----------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation/special chars
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# ---------- Load model ----------
model = SentenceTransformer('all-MiniLM-L6-v2')

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
    clean1 = preprocess_text(req.text1)
    clean2 = preprocess_text(req.text2)
    score = get_similarity(clean1, clean2)
    return {"similarity score": score}
