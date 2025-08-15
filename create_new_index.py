#!/usr/bin/env python3
"""
Create a new high-quality Pinecone index with larger dimensions
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Delete old index if exists
try:
    pc.delete_index("final-index")
    print("Deleted old index")
except:
    print("Old index not found or already deleted")

# Create new high-quality index
print("Creating new high-quality index...")
pc.create_index(
    name="final-index",
    dimension=3072,  # text-embedding-3-large dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

print("✅ New high-quality index created successfully!")
print("Now run: python embedding.py")