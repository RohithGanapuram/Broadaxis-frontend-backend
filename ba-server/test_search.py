#!/usr/bin/env python3
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from pinecone import Pinecone
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("final-index")

def _embed_text(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def test_project_coordinator_search():
    query = "detailed job duties of project coordinator"
    print(f"Query: {query}")
    
    query_embedding = _embed_text(query)
    
    result = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
        namespace="broadaxis-index"
    )
    
    if hasattr(result, 'matches'):
        matches = result.matches
        print(f"Found {len(matches)} matches")
        
        for i, match in enumerate(matches):
            print(f"\nMatch {i+1}:")
            print(f"  Score: {match.score:.4f}")
            if match.metadata:
                text = match.metadata.get('text', '')
                # Clean up special characters for display
                clean_text = text.replace('\uf0b7', '•').replace('\u2022', '•')
                print(f"  Text length: {len(clean_text)}")
                print(f"  Text preview: {clean_text[:300]}...")

if __name__ == "__main__":
    test_project_coordinator_search()