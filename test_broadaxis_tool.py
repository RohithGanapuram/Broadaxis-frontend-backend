#!/usr/bin/env python3
"""
Test script for BroadAxis knowledge search tool
This script directly tests the MCP interface and tool execution
"""

import asyncio
import json
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from mcp_interface import run_mcp_query

async def test_broadaxis_query():
    """Test the BroadAxis knowledge search with a direct query"""
    
    print("🧪 Testing BroadAxis Knowledge Search Tool")
    print("=" * 50)
    
    # Test query
    test_query = "tell me about broadaxis and where are they based on"
    print(f"📝 Test Query: {test_query}")
    print()
    
    try:
        print("🔄 Calling MCP interface...")
        
        # Call the MCP interface directly (no WebSocket)
        result = await run_mcp_query(
            query=test_query,
            enabled_tools=None,  # Use all available tools
            model="claude-3-7-sonnet-20250219",
            session_id="test_session",
            websocket=None,  # No WebSocket for direct testing
            send_message_callback=None
        )
        
        print("✅ MCP interface call completed")
        print()
        
        # Display results
        print("📊 Results:")
        print("-" * 30)
        
        if isinstance(result, dict) and "response" in result:
            print(f"Response: {result['response']}")
            print()
            
            if "tokens_used" in result:
                print(f"Tokens used: {result['tokens_used']}")
        else:
            print(f"Unexpected result format: {type(result)}")
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

async def test_tool_listing():
    """Test if we can list available tools"""
    
    print("\n🔧 Testing Tool Listing")
    print("=" * 30)
    
    try:
        from mcp_interface import mcp_interface
        
        print("🔄 Initializing MCP interface...")
        await mcp_interface.initialize()
        
        print("📋 Available Tools:")
        tools = await mcp_interface.get_tools()
        
        for i, tool in enumerate(tools, 1):
            print(f"  {i}. {tool['name']} - {tool['description']}")
            
        print(f"\nTotal tools available: {len(tools)}")
        
        # Check if Broadaxis_knowledge_search is available
        broadaxis_tool = next((tool for tool in tools if tool['name'] == 'Broadaxis_knowledge_search'), None)
        if broadaxis_tool:
            print("✅ Broadaxis_knowledge_search tool is available")
        else:
            print("❌ Broadaxis_knowledge_search tool NOT found!")
            
    except Exception as e:
        print(f"❌ Error listing tools: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

async def test_direct_tool_call():
    """Test calling the BroadAxis tool directly"""
    
    print("\n🎯 Testing Direct Tool Call")
    print("=" * 30)
    
    try:
        from mcp_interface import mcp_interface
        
        print("🔄 Initializing MCP interface...")
        await mcp_interface.initialize()
        
        print("🔍 Calling Broadaxis_knowledge_search directly...")
        
        # Call the tool directly
        result = await mcp_interface.session.call_tool(
            "Broadaxis_knowledge_search",
            arguments={
                "query": "BroadAxis company information location headquarters",
                "top_k": 5,
                "min_score": 0.2
            }
        )
        
        print("✅ Direct tool call completed")
        print(f"📄 Result: {result.content}")
        
    except Exception as e:
        print(f"❌ Error in direct tool call: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

async def main():
    """Run all tests"""
    
    print("🚀 Starting BroadAxis Tool Tests")
    print("=" * 50)
    
    # Test 1: List available tools
    await test_tool_listing()
    
    # Test 2: Direct tool call
    await test_direct_tool_call()
    
    # Test 3: Full MCP interface test
    await test_broadaxis_query()
    
    print("\n🏁 All tests completed")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
