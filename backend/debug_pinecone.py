#!/usr/bin/env python3
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

try:
    from pinecone import Pinecone
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pinecone_client.Index("final-index")
    
    print("=== PINECONE DEBUG ===")
    
    # Get index stats
    stats = pinecone_index.describe_index_stats()
    print(f"Index stats: {stats}")
    
    # List all namespaces
    print(f"\nNamespaces found:")
    for namespace, info in stats.namespaces.items():
        print(f"  - {namespace}: {info['vector_count']} vectors")
    
    # Test query in different namespaces
    test_namespaces = ['15-Specifications', '5-General Requirements for Proposals', 'broadaxis-index']
    
    for namespace in test_namespaces:
        print(f"\n=== Testing namespace: {namespace} ===")
        try:
            result = pinecone_index.query(
                vector=[0.0] * 1536,
                top_k=3,
                include_metadata=True,
                namespace=namespace
            )
            print(f"Found {len(result.matches)} vectors")
            for match in result.matches[:2]:
                print(f"  - ID: {match.id}")
                print(f"    Text preview: {match.metadata.get('text', 'No text')[:100]}...")
        except Exception as e:
            print(f"Error querying {namespace}: {e}")

except Exception as e:
    print(f"Error: {e}")