import os
import re
import subprocess
import numpy as np
import faiss
import spacy
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# -----------------------------
# 🔹 Environment setup
# -----------------------------
load_dotenv()
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

HF_TOKEN = os.getenv("HF_TOKEN")

# -----------------------------
# 🔹 Lazy spaCy loader (saves memory)
# -----------------------------
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_person_name(text: str):
    """Detect the first PERSON entity (name) using spaCy."""
    nlp = get_nlp()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None


# -----------------------------
# 🔹 Lazy FAISS & embedding loader
# -----------------------------
INDEX_PATH = "vectorstores/db_faiss/index.faiss"
TEXTS_PATH = "vectorstores/db_faiss/texts.npy"

_embedding_model = None
_index = None
_texts = None

def load_faiss_resources():
    """Load embedding model, FAISS index, and texts only once."""
    global _embedding_model, _index, _texts
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if _texts is None:
        _texts = np.load(TEXTS_PATH, allow_pickle=True)
        print(f"✅ Loaded knowledge base with {_texts.shape[0]} text chunks.")
    return _embedding_model, _index, _texts


def search_knowledge_base(query: str, top_k: int = 30):
    embedding_model, index, texts = load_faiss_resources()
    q_emb = embedding_model.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = [texts[i] for i in I[0]]
    return "\n".join(results)


# -----------------------------
# 🔹 Hugging Face client
# -----------------------------
hf_client = InferenceClient(token=HF_TOKEN)


# -----------------------------
# 🔹 Main response generator
# -----------------------------
def get_response(user_input: str, chat_history=None, last_person=None):
    person = extract_person_name(user_input)

    # Track last mentioned person
    if not person and last_person:
        if re.search(r"\b(he|him|his|she|her|hers|they|them|their)\b", user_input, re.I):
            person = last_person
            user_input = f"{user_input} (referring to {person})"

    # Retrieve context from FAISS
    context = search_knowledge_base(user_input)
    conversation_context = chat_history if chat_history else "No previous conversation."

    prompt = f"""
You are a helpful HR assistant.
Use the conversation history and HR knowledge base below to answer clearly and concisely.

Conversation Context:
{conversation_context}

Knowledge Base Context:
{context}

Question: {user_input}

Answer:
"""

    try:
        response = hf_client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[
                {"role": "system", "content": "You are a helpful HR assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error from Hugging Face API: {str(e)}"
