# ingest_single_pdf.py
# pip install openai pinecone-client python-dotenv PyPDF2 langchain-text-splitters

import os
import sys
import hashlib
import re
from typing import List, Iterable

from dotenv import load_dotenv
load_dotenv()

# --- OpenAI ---
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"   # 1536-dim

# --- Pinecone v3 ---
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "final-index"
INDEX_DIM = 1536  # must match text-embedding-3-small

# --- PDF & chunking ---
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ensure_index(name: str = INDEX_NAME, dimension: int = INDEX_DIM, metric: str = "cosine"):
    existing = {i["name"] for i in pc.list_indexes()}
    if name not in existing:
        print(f"[pinecone] creating index '{name}' (dim={dimension}, metric={metric})...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"[pinecone] index '{name}' exists.")
    return pc.Index(name)


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts)


# Optimized splitter for high-quality embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,            # Optimal size for embedding quality
    chunk_overlap=300,          # Preserve context boundaries
    separators=[
        "\n\n\n",              # Major section breaks
        "Position Title:",       # Job position markers
        "Detailed Job Duties:",  # Job duties sections
        "Key Highlights:",       # Highlight sections
        "\n\n",                 # Paragraph breaks
        "\n",                   # Line breaks
        ". ",                   # Sentence breaks
        " "                     # Word breaks
    ],
)


def _batched(xs: Iterable, size: int) -> Iterable[List]:
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """Create high-quality embeddings with preprocessing"""
    result: List[List[float]] = []
    
    # Preprocess texts for better embedding quality
    processed_texts = []
    for text in texts:
        # Clean and normalize text
        clean_text = text.strip()
        # Remove excessive whitespace but preserve structure
        clean_text = re.sub(r'\s+', ' ', clean_text)
        # Add context markers for better semantic understanding
        if "Position Title:" in clean_text:
            clean_text = f"Job Role Information: {clean_text}"
        elif "Detailed Job Duties:" in clean_text:
            clean_text = f"Job Responsibilities: {clean_text}"
        elif any(keyword in clean_text.lower() for keyword in ["service", "capability", "expertise"]):
            clean_text = f"Company Services: {clean_text}"
        
        processed_texts.append(clean_text)
    
    # Create embeddings in smaller batches for better quality
    for batch in _batched(processed_texts, batch_size):
        resp = client.embeddings.create(
            model=EMBED_MODEL, 
            input=batch
            # Uses model default dimensions (3072 for text-embedding-3-large)
        )
        result.extend([d.embedding for d in resp.data])
    
    return result


def make_id(pdf_basename: str, chunk_text: str, i: int) -> str:
    """Create semantic IDs based on content type"""
    h = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()[:12]
    
    # Add semantic prefixes for better organization
    if "Position Title:" in chunk_text:
        return f"{pdf_basename}_job_{i}_{h}"
    elif "service" in chunk_text.lower() or "capability" in chunk_text.lower():
        return f"{pdf_basename}_service_{i}_{h}"
    elif "experience" in chunk_text.lower() or "project" in chunk_text.lower():
        return f"{pdf_basename}_experience_{i}_{h}"
    else:
        return f"{pdf_basename}_content_{i}_{h}"


def process_pdf_and_upsert(pdf_path: str, index, namespace: str = ""):
    basename = os.path.basename(pdf_path)
    print(f"[pdf] extracting text: {basename}")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"[skip] no extractable text in {basename}")
        return

    print(f"[split] chunking text...")
    chunks = text_splitter.split_text(text)
    if not chunks:
        print(f"[skip] produced 0 chunks for {basename}")
        return

    print(f"[embed] embedding {len(chunks)} chunks...")
    vectors = embed_texts(chunks)

    print(f"[upsert] sending to Pinecone: {len(vectors)} vectors")
    payload = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        payload.append({
            "id": make_id(basename, chunk, i),
            "values": vec,
            "metadata": {"source": basename, "text": chunk}
        })

    for batch in _batched(payload, 100):
        index.upsert(vectors=batch, namespace=namespace)

    print(f"[done] upserted {len(payload)} chunks from {basename}")


def main():
    # Your exact Windows path:
    pdf_path = r"C:\Users\rohka\OneDrive\Desktop\ba-rfpapp\infoDoc.pdf"
    if not os.path.isfile(pdf_path):
        raise SystemExit(f"PDF not found: {pdf_path}")

    index = ensure_index(INDEX_NAME, INDEX_DIM, "cosine")
    process_pdf_and_upsert(pdf_path, index=index, namespace="broadaxis-index")  # set namespace if you want


if __name__ == "__main__":
    main()
