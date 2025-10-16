from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
import numpy as np
import os


# === Paths ===
DATA_PATH = "D:/tutorial/data/General HR Queries.pdf"
DB_FAISS_PATH = "D:/tutorial/vectorstores/db_faiss"
EMPLOYEE_CSV_PATH = "D:/tutorial/data/employees.csv"

# === Step 1: Load your HR document ===
loader = UnstructuredPDFLoader(DATA_PATH, mode="elements")
documents = loader.load()

# === Step 2: Split text into chunks ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=2000,
    separators=["\n\n", "\n", ".", "!", "?", ";"]
)
texts = text_splitter.split_documents(documents)
print(f" Split into {len(texts)} text chunks")
# === Step 3: Create embeddings ===
embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)
db.save_local(DB_FAISS_PATH)


# === Step 4: Create FAISS vector store ===
db = FAISS.from_documents(texts, embeddings)

# === Step 5: Save FAISS database locally ===
db.save_local(DB_FAISS_PATH)

# === Step 6: Save raw text chunks (for later retrieval) ===
texts_list = [t.page_content for t in texts]
np.save(os.path.join(DB_FAISS_PATH, "texts.npy"), texts_list)

print("✅ FAISS Vector DB and texts saved successfully at:", DB_FAISS_PATH)
print(f"✅ Total chunks stored: {len(texts_list)}")
