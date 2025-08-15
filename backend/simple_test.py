#!/usr/bin/env python3
import os
import sys
import asyncio
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from api import create_smart_chunks, create_embeddings, store_chunks_in_pinecone, openai_client, pinecone_index

def test_chunking():
    print("=== CHUNKING TEST ===")
    sample_text = "This is paragraph one.\n\nThis is paragraph two with more content to test chunking.\n\nThis is paragraph three." * 20
    print(f"Text length: {len(sample_text)}")
    
    chunks = create_smart_chunks(sample_text, "test.pdf")
    print(f"Chunks created: {len(chunks)}")
    return chunks

async def test_embeddings():
    print("\n=== EMBEDDINGS TEST ===")
    if not openai_client:
        print("OpenAI client not available")
        return None
    
    texts = ["Test text one", "Test text two"]
    embeddings = await create_embeddings(texts)
    print(f"Embeddings created: {len(embeddings) if embeddings else 0}")
    return embeddings

async def test_pinecone():
    print("\n=== PINECONE TEST ===")
    if not pinecone_index:
        print("Pinecone not available")
        return False
    
    chunks = [{"text": "test", "filename": "test.pdf", "chunk_index": 0}]
    embeddings = await create_embeddings(["test"])
    
    if embeddings:
        success = await store_chunks_in_pinecone(chunks, embeddings, "test.pdf")
        print(f"Storage success: {success}")
        return success
    return False

async def main():
    chunks = test_chunking()
    embeddings = await test_embeddings()
    pinecone_ok = await test_pinecone()
    
    print(f"\nRESULTS:")
    print(f"Chunking: {'OK' if chunks else 'FAIL'}")
    print(f"Embeddings: {'OK' if embeddings else 'FAIL'}")
    print(f"Pinecone: {'OK' if pinecone_ok else 'FAIL'}")

if __name__ == "__main__":
    asyncio.run(main())