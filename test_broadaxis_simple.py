#!/usr/bin/env python3
"""
Simple test script for BroadAxis knowledge search tool with detailed debugging
"""

import asyncio
import sys
import os
import time

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_direct_tool_call():
    """Test calling the BroadAxis tool directly with timeout"""
    
    print("🎯 Testing Direct BroadAxis Tool Call")
    print("=" * 40)
    
    try:
        from mcp_interface import mcp_interface
        
        print("🔄 Step 1: Initializing MCP interface...")
        start_time = time.time()
        await mcp_interface.initialize()
        print(f"✅ MCP interface initialized in {time.time() - start_time:.2f}s")
        
        print("\n🔍 Step 2: Calling Broadaxis_knowledge_search directly...")
        print("Query: 'BroadAxis company information location headquarters'")
        
        # Add timeout to the tool call
        start_time = time.time()
        
        try:
            # Call the tool with a timeout
            result = await asyncio.wait_for(
                mcp_interface.session.call_tool(
                    "Broadaxis_knowledge_search",
                    arguments={
                        "query": "BroadAxis company information location headquarters",
                        "top_k": 3,  # Reduced for faster testing
                        "min_score": 0.2
                    }
                ),
                timeout=60.0  # 60 second timeout
            )
            
            print(f"✅ Direct tool call completed in {time.time() - start_time:.2f}s")
            print(f"📄 Result type: {type(result)}")
            print(f"📄 Result content: {result.content}")
            
        except asyncio.TimeoutError:
            print(f"❌ Tool call timed out after {time.time() - start_time:.2f}s")
            print("The BroadAxis tool is hanging during execution")
            
        except Exception as tool_error:
            print(f"❌ Tool call failed after {time.time() - start_time:.2f}s")
            print(f"Error: {str(tool_error)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        
    except Exception as e:
        print(f"❌ Error in setup: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

async def test_pinecone_connection():
    """Test Pinecone connection directly"""
    
    print("\n🌲 Testing Pinecone Connection")
    print("=" * 30)
    
    try:
        # Import the server module to test Pinecone directly
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ba-server'))
        
        # Test Pinecone connection
        from pinecone import Pinecone
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
        
        pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        if not pinecone_api_key:
            print("❌ PINECONE_API_KEY not found in environment")
            return
            
        print("🔄 Connecting to Pinecone...")
        start_time = time.time()
        
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("final-index")
        
        print("🔄 Getting index stats...")
        stats = index.describe_index_stats()
        print(f"✅ Pinecone connected in {time.time() - start_time:.2f}s")
        print(f"📊 Index stats: {stats}")
        
        # Check for broadaxis-index namespace
        namespaces = stats.get('namespaces', {})
        if 'broadaxis-index' in namespaces:
            vector_count = namespaces['broadaxis-index'].get('vector_count', 0)
            print(f"✅ Found {vector_count} vectors in 'broadaxis-index' namespace")
        else:
            print("❌ 'broadaxis-index' namespace not found!")
            print(f"Available namespaces: {list(namespaces.keys())}")
            
    except Exception as e:
        print(f"❌ Pinecone connection failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

async def test_embedding():
    """Test OpenAI embedding creation"""
    
    print("\n🧠 Testing OpenAI Embedding")
    print("=" * 30)
    
    try:
        from openai import OpenAI
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
        
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return
            
        print("🔄 Creating OpenAI client...")
        client = OpenAI(api_key=openai_api_key)
        
        test_text = "BroadAxis company information location headquarters"
        print(f"🔄 Creating embedding for: '{test_text}'")
        
        start_time = time.time()
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[test_text]
        )
        embedding = resp.data[0].embedding
        
        print(f"✅ Embedding created in {time.time() - start_time:.2f}s")
        print(f"📊 Embedding dimension: {len(embedding)}")
        
    except Exception as e:
        print(f"❌ Embedding creation failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

async def main():
    """Run all tests"""
    
    print("🚀 Starting BroadAxis Debug Tests")
    print("=" * 50)
    
    # Test 1: Pinecone connection
    await test_pinecone_connection()
    
    # Test 2: OpenAI embedding
    await test_embedding()
    
    # Test 3: Direct tool call
    await test_direct_tool_call()
    
    print("\n🏁 All tests completed")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
