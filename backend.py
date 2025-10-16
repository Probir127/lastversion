import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import spacy
import re

nlp = spacy.load("en_core_web_sm")

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

def extract_person_name(text):
    """
    Automatically detect the first PERSON entity (e.g., names)
    from the user's message using spaCy.
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

# Load environment variables
load_dotenv()

# Configure Hugging Face Inference Client
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))

# --- ✅ Load Embedding Model & FAISS Index ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = "D:/tutorial/vectorstores/db_faiss/index.faiss"
TEXTS_PATH = "D:/tutorial/vectorstores/db_faiss/texts.npy"

index = faiss.read_index(INDEX_PATH)
texts = np.load(TEXTS_PATH, allow_pickle=True)
print(f"✅ Loaded knowledge base with {len(texts)} text chunks.")

def search_knowledge_base(query, top_k=30):
    q_emb = embedding_model.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = [texts[i] for i in I[0]]
    return "\n".join(results)

def get_response(user_input, chat_history=None, last_person=None):
    person = extract_person_name(user_input)

    if not person and last_person:
        if re.search(r"\b(he|him|his|she|her|hers|they|them|their)\b", user_input, re.I):
            person = last_person
            user_input = f"{user_input} (referring to {person})"

    context = search_knowledge_base(user_input)
    conversation_context = chat_history if chat_history else "No previous conversation."

    prompt = f"""You are a helpful HR assistant.
Use the previous conversation context and knowledge base below to answer clearly and concisely.

Conversation Context:
{conversation_context}

Knowledge Base Context:
{context}

Question: {user_input}

Answer:"""

    try:
        # ✅ Use conversational task instead of text-generation
        response = hf_client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[
                {"role": "system", "content": "You are a helpful HR assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message["content"]

    except Exception as e:
        return f"Error from Hugging Face API: {str(e)}"
