#!/usr/bin/env python3
"""
Test script to verify chunking and vector storage pipeline
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Test chunking
def test_chunking():
    print("=== TESTING CHUNKING ===")
    
    # Import the chunking function
    from api import create_smart_chunks, CHUNK_SIZE
    
    # Test with sample text
    sample_text = """This is a test document with multiple paragraphs.

This is the second paragraph that should help us understand how the chunking works.

This is the third paragraph. We want to see if the system properly splits long documents into manageable chunks.

This is the fourth paragraph. The chunking system should create multiple chunks when the text exceeds the chunk size limit.

This is the fifth paragraph. Each chunk should maintain semantic boundaries while staying under the size limit.

This is the sixth paragraph. The overlap between chunks helps maintain context across boundaries.

This is the seventh paragraph. We need to verify that all chunks are properly created and indexed.

This is the eighth paragraph. The system should handle various document formats and structures.

This is the ninth paragraph. Proper chunking is essential for accurate vector search and retrieval.

This is the tenth paragraph. Let's see how many chunks this creates with our current settings."""
    
    print(f"Sample text length: {len(sample_text)} characters")
    print(f"Chunk size setting: {CHUNK_SIZE}")
    
    chunks = create_smart_chunks(sample_text, "test_document.txt")
    
    print(f"\nChunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {len(chunk['text'])} chars, index: {chunk['chunk_index']}")
        print(f"Preview: {chunk['text'][:100]}...")
        print()
    
    return chunks

# Test embeddings
async def test_embeddings():
    print("=== TESTING EMBEDDINGS ===")
    
    from api import create_embeddings, openai_client
    
    if not openai_client:
        print("❌ OpenAI client not available")
        return None
    
    test_texts = [
        "This is a test document about RFP requirements.",
        "The second chunk discusses technical specifications.",
        "The third chunk covers budget and timeline information."
    ]
    
    print(f"Creating embeddings for {len(test_texts)} texts...")
    embeddings = await create_embeddings(test_texts)
    
    if embeddings:
        print(f"✅ Created {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        return embeddings
    else:
        print("❌ Failed to create embeddings")
        return None

# Test Pinecone storage
async def test_pinecone():
    print("=== TESTING PINECONE ===")
    
    from api import pinecone_index, store_chunks_in_pinecone
    
    if not pinecone_index:
        print("❌ Pinecone not available")
        return False
    
    # Create test chunks
    test_chunks = [
        {"text": "Test chunk 1", "filename": "test.pdf", "chunk_index": 0},
        {"text": "Test chunk 2", "filename": "test.pdf", "chunk_index": 1}
    ]
    
    # Create test embeddings
    embeddings = await create_embeddings([chunk["text"] for chunk in test_chunks])
    
    if not embeddings:
        print("❌ Cannot test Pinecone without embeddings")
        return False
    
    # Test storage
    print("Storing test chunks in Pinecone...")
    success = await store_chunks_in_pinecone(test_chunks, embeddings, "test.pdf")
    
    if success:
        print("✅ Successfully stored chunks in Pinecone")
        
        # Test retrieval
        print("Testing retrieval...")
        try:
            namespace = "test"
            results = pinecone_index.query(
                vector=embeddings[0],
                top_k=2,
                include_metadata=True,
                namespace=namespace
            )
            print(f"✅ Retrieved {len(results.matches)} results")
            for match in results.matches:
                print(f"  Score: {match.score:.3f}, Text: {match.metadata.get('text', 'N/A')[:50]}...")
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
        
        return True
    else:
        print("❌ Failed to store chunks in Pinecone")
        return False

# Main test function
async def main():
    print("VECTOR PIPELINE TEST\n")
    
    # Test 1: Chunking
    chunks = test_chunking()
    
    # Test 2: Embeddings
    embeddings = await test_embeddings()
    
    # Test 3: Pinecone
    pinecone_success = await test_pinecone()
    
    print("\n=== SUMMARY ===")
    print(f"Chunking: {'✅ Working' if chunks else '❌ Failed'}")
    print(f"Embeddings: {'✅ Working' if embeddings else '❌ Failed'}")
    print(f"Pinecone: {'✅ Working' if pinecone_success else '❌ Failed'}")
    
    if chunks and embeddings and pinecone_success:
        print("\n🎉 All systems working correctly!")
    else:
        print("\n⚠️ Some components need attention")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())