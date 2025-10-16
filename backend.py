import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import re

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

def extract_person_name(text):
    """Simple name extraction without spaCy"""
    words = text.split()
    for i, word in enumerate(words):
        if word and word[0].isupper() and i > 0 and len(word) > 1 and word.isalpha():
            return word
    return None

load_dotenv()
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "vectorstores", "db_faiss", "index.faiss")
TEXTS_PATH = os.path.join(BASE_DIR, "vectorstores", "db_faiss", "texts.npy")

try:
    index = faiss.read_index(INDEX_PATH)
    texts = np.load(TEXTS_PATH, allow_pickle=True)
    print(f"✅ Loaded knowledge base with {len(texts)} text chunks.")
    FAISS_LOADED = True
except Exception as e:
    print(f"⚠️ Warning: Could not load FAISS database: {e}")
    index = None
    texts = None
    FAISS_LOADED = False

def search_knowledge_base(query, top_k=30):
    if not FAISS_LOADED:
        return "No knowledge base available."
    try:
        q_emb = embedding_model.encode([query])
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, top_k)
        results = [texts[i] for i in I[0]]
        return "\n".join(results)
    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        return "Error searching knowledge base."

def get_response(user_input, chat_history=None, last_person=None):
    person = extract_person_name(user_input)
    
    if not person and last_person:
        if re.search(r"\b(he|him|his|she|her|hers|they|them|their)\b", user_input, re.I):
            person = last_person
            user_input = f"{user_input} (referring to {person})"
    
    context = search_knowledge_base(user_input) if FAISS_LOADED else "No knowledge base available."
    conversation_context = chat_history if chat_history else "No previous conversation."
    
    prompt = f"""You are a helpful HR assistant.
Use the previous conversation context and knowledge base below to answer clearly and concisely.

Conversation Context:
{conversation_context}

Knowledge Base Context:
{context}

Question: {user_input}

Answer (be helpful, clear, and concise):"""
    
    try:
        response = hf_client.text_generation(
            prompt,
            model="microsoft/Phi-3-mini-4k-instruct",
            max_new_tokens=512,
            temperature=0.7,
        )
        return response
    except Exception as e:
        error_msg = str(e)
        print(f"Error from Hugging Face API: {error_msg}")
        if "rate limit" in error_msg.lower():
            return "Sorry, the API rate limit has been reached. Please try again in a few moments."
        elif "token" in error_msg.lower():
            return "API authentication error. Please check the configuration."
        else:
            return f"I apologize, but I'm having trouble connecting to the AI service right now. Please try again later."

if __name__ == "__main__":
    print("\n🧪 Testing backend...")
    print(get_response("How do I apply for parental leave?"))


