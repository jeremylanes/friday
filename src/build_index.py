"""
build_index.py - Index Marvel with Cohere embeddings (free tier) + FAISS
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

load_dotenv()

DATA_DIR = "data"
DB_DIR = "db"
os.makedirs(DB_DIR, exist_ok=True)

def build_index():
    print("🔨 Building Marvel index with Cohere...")

    files = ["mcu_facts.txt", "comics_facts.txt"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_docs = []

    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"📖 Reading {path}")

        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()

        chunks = splitter.split_documents(docs)

        all_docs.extend(chunks)

    print("🧠 Generating embeddings (Cohere)...")
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    db = FAISS.from_documents(all_docs, embeddings)
    db.save_local(os.path.join(DB_DIR, "marvel_index"))

    print("✅ Index saved → db/marvel_index")

if __name__ == "__main__":
    build_index()
