import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- PDF Parsing ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Chunking ---
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# --- Embedding Model ---
class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Use float32 for embeddings (default and safest for most models)
        self.model = SentenceTransformer(model_name)
    def encode(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

# --- FAISS Vector DB ---
class VectorDB:
    def __init__(self, dim, db_path="faiss.index"):
        self.index = faiss.IndexFlatL2(dim)
        self.db_path = db_path
        self.chunks = []
    def add(self, embeddings, chunks):
        # Store and index embeddings as float32 (default for FAISS)
        emb_f32 = np.array(embeddings).astype('float32')
        self.index.add(emb_f32)
        self.chunks.extend(chunks)
    def search(self, query_emb, top_k=2):
        D, I = self.index.search(np.array([query_emb]).astype('float32'), top_k)
        return [self.chunks[i] for i in I[0] if i < len(self.chunks)]
    def save(self):
        faiss.write_index(self.index, self.db_path)
    def load(self):
        if os.path.exists(self.db_path):
            self.index = faiss.read_index(self.db_path)

# --- Utility: End-to-end PDF to FAISS ---
def process_pdf_to_faiss(pdf_path, embed_model, vectordb):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    # Filter out empty chunks
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        return []
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings)
    if len(embeddings.shape) == 1:
        # Only one chunk, reshape to (1, d)
        embeddings = embeddings.reshape(1, -1)
    vectordb.add(embeddings, chunks)
    vectordb.save()
    return chunks
