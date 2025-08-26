"""
MCP Interface for BroadAxis RFP/RFQ Management Platform
"""

import asyncio
import json
import os
from typing import Dict, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from error_handler import BroadAxisError, ExternalAPIError, error_handler


class MCPInterface:
    def __init__(self):
        self.server_params = StdioServerParameters(
            command='python',
            args=[os.path.join(os.path.dirname(__file__), "..", "ba-server", "server.py")],
            env=None,
        )
        self.anthropic = None
        self._tools_cache = None
        self._prompts_cache = None
        self._cache_lock = asyncio.Lock()
        self._connection_status = "disconnected"
        self._initializing = False
        
        # Persistent connection management
        self.session = None
        self.stdio = None
        self.write = None
        self.exit_stack = None
        
        try:
            from anthropic import AsyncAnthropic
            self.anthropic = AsyncAnthropic()
        except ImportError:
            print("Warning: Anthropic not available")
        
    async def _ensure_connection(self):
        """Ensure we have an active connection to the MCP server"""
        if self.session is None or self._connection_status != "connected":
            await self.initialize()
    
    async def initialize(self):
        """Initialize connection to MCP server and fetch tools/prompts"""
        async with self._cache_lock:
            if self._tools_cache is None or self._prompts_cache is None or self.session is None:
                self._initializing = True
                try:
                    self._connection_status = "connecting"
                    
                    # Create persistent connection
                    if self.exit_stack is None:
                        self.exit_stack = AsyncExitStack()
                    
                    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(self.server_params))
                    self.stdio, self.write = stdio_transport
                    self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
                    
                    await self.session.initialize()
                    self._connection_status = "connected"
                    
                    # Fetch both tools and prompts in parallel
                    tools_task = self.session.list_tools()
                    prompts_task = self.session.list_prompts()
                    
                    tools_response, prompts_response = await asyncio.gather(
                        tools_task, prompts_task, return_exceptions=True
                    )
                    
                    # Process tools response
                    if isinstance(tools_response, Exception):
                        raise tools_response
                    if not tools_response or not hasattr(tools_response, 'tools'):
                        raise ExternalAPIError("Invalid tools response from MCP server")
                    
                    self._tools_cache = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    } for tool in tools_response.tools]
                    
                    # Process prompts response
                    if isinstance(prompts_response, Exception):
                        raise prompts_response
                    if not prompts_response or not hasattr(prompts_response, 'prompts'):
                        raise ExternalAPIError("Invalid prompts response from MCP server")
                    
                    self._prompts_cache = [{
                        "name": prompt.name,
                        "description": prompt.description
                    } for prompt in prompts_response.prompts]
                    
                except Exception as e:
                    self._connection_status = "offline"
                    self._tools_cache = self._tools_cache or []
                    self._prompts_cache = self._prompts_cache or []
                    
                    if isinstance(e, BroadAxisError):
                        raise e
                    else:
                        error_handler.log_error(e, {'operation': 'initialize_mcp'})
                        raise ExternalAPIError("Failed to initialize MCP server", 
                                             {'original_error': str(e)})
                finally:
                    self._initializing = False
    
    async def get_tools(self):
        """Get cached tools, initialize if needed"""
        if self._tools_cache is None:
            await self.initialize()
        return self._tools_cache
    
    async def get_prompts(self):
        """Get cached prompts, initialize if needed"""
        if self._prompts_cache is None:
            await self.initialize()
        return self._prompts_cache
    
    async def process_query_with_anthropic(self, query: str, enabled_tools: List[str] = None, model: str = None, session_id: str = "default", websocket = None, send_message_callback=None) -> Dict:
        if not self.anthropic:
            return {"response": "Anthropic API not available", "tokens_used": 0}
        
        try:
            # Ensure we have an active connection
            await self._ensure_connection()
            
            available_tools = await self.get_tools()
            
            # Filter tools based on query context
            if enabled_tools:
                available_tools = [tool for tool in available_tools if tool["name"] in enabled_tools]
            
            system_prompt = "I'm BroadAxis AI. For company questions, I search our knowledge base first. For market research, I use web search. I provide direct answers for general conversation."
            
            selected_model = model or "claude-3-7-sonnet-20250219"
            
            # Build messages for the API call
            messages = [{"role": "user", "content": query}]
            
            # Simple API call without retry logic
            try:
                response = await self.anthropic.messages.create(
                    max_tokens=4096,
                    model=selected_model,
                    system=system_prompt,
                    tools=available_tools,
                    messages=messages
                )
                
                # Send progress update for initial processing
                if websocket and send_message_callback:
                    await send_message_callback(
                        json.dumps({
                            "type": "progress",
                            "message": "Processing your request...",
                            "progress": 30,
                            "step": "processing",
                            "current_step": 1,
                            "total_steps": 1
                        }),
                        websocket
                    )
                
            except Exception as api_error:
                return {"response": f"❌ **API Error: {str(api_error)}**", "tokens_used": 0}
            
            full_response = ""
            tools_used = []
            
            process_query = True
            while process_query:
                assistant_content = []
                tool_calls = []
                
                # Collect all content and tool calls
                for content in response.content:
                    if content.type == 'text':
                        full_response += content.text
                        assistant_content.append(content)
                        if len(response.content) == 1:
                            process_query = False
                    elif content.type == 'tool_use':
                        tools_used.append(content.name)
                        assistant_content.append(content)
                        tool_calls.append(content)
                
                # Execute all tool calls in parallel if any exist
                if tool_calls:
                    messages.append({'role': 'assistant', 'content': assistant_content})
                    
                    # Send progress update for tool execution
                    if websocket and send_message_callback:
                        tool_names = [tool.name for tool in tool_calls]
                        progress_message = f"Executing {len(tool_calls)} tools: {', '.join(tool_names[:2])}"
                        if len(tool_names) > 2:
                            progress_message += f" and {len(tool_names) - 2} more..."
                        
                        await send_message_callback(
                            json.dumps({
                                "type": "progress",
                                "message": progress_message,
                                "progress": 0,  # Start at 0, will be updated as tools complete
                                "step": "tool_execution",
                                "current_step": 1,
                                "total_steps": len(tool_calls)
                            }),
                            websocket
                        )
                    
                    # Simple tool execution without token management
                    tool_sem = asyncio.Semaphore(2)  # Allow 2 concurrent tools
                    _inflight: Dict[tuple, asyncio.Task] = {}

                    async def execute_tool(tool_content):
                        key = (tool_content.name, json.dumps(tool_content.input, sort_keys=True))
                        if key in _inflight:
                            return await _inflight[key]

                        async def _run():
                            async with tool_sem:
                                try:
                                    # Use the persistent session instead of creating a new one
                                    result = await self.session.call_tool(tool_content.name, arguments=tool_content.input)
                                    return {
                                        "type": "tool_result",
                                        "tool_use_id": tool_content.id,
                                        "content": result.content
                                    }
                                except Exception as tool_error:
                                    error_handler.log_error(tool_error, {'tool_name': tool_content.name})
                                    return {
                                        "type": "tool_result",
                                        "tool_use_id": tool_content.id,
                                        "content": [{"type": "text", "text": f"Tool failed: {str(tool_error)}"}]
                                    }

                        task = asyncio.create_task(_run())
                        _inflight[key] = task
                        try:
                            return await task
                        finally:
                            _inflight.pop(key, None)
                    
                    # Execute all tools concurrently with progress updates
                    completed_tools = 0
                    total_tools = len(tool_calls)
                    
                    async def execute_tool_with_progress(tool):
                        nonlocal completed_tools
                        result = await execute_tool(tool)
                        completed_tools += 1
                        
                        # Send progress update
                        if websocket and send_message_callback:
                            progress_percentage = int((completed_tools / total_tools) * 100)
                            await send_message_callback(
                                json.dumps({
                                    "type": "progress",
                                    "message": f"Completed {completed_tools}/{total_tools} tools",
                                    "progress": progress_percentage,
                                    "step": "tool_execution",
                                    "current_step": completed_tools,
                                    "total_steps": total_tools
                                }),
                                websocket
                            )
                        return result
                    
                    tool_results = await asyncio.gather(
                        *[execute_tool_with_progress(tool) for tool in tool_calls],
                        return_exceptions=True
                    )
                    
                    # Batch all tool results into a single user message
                    batched_tool_results = []
                    for result in tool_results:
                        if isinstance(result, dict):
                            batched_tool_results.append(result)
                        else:
                            error_handler.log_error(result, {'operation': 'parallel_tool_execution'})
                    
                    if batched_tool_results:
                        messages.append({"role": "user", "content": batched_tool_results})
                    
                    # --- Simple follow-up call without retry logic ---
                    # Send progress update for response generation
                    if websocket and send_message_callback:
                        await send_message_callback(
                            json.dumps({
                                "type": "progress",
                                "message": "Generating final response...",
                                "progress": 90,
                                "step": "generation",
                                "current_step": 1,
                                "total_steps": 1
                            }),
                            websocket
                        )

                    try:
                        response = await self.anthropic.messages.create(
                            max_tokens=2048,
                            model=selected_model,
                            system=system_prompt,
                            tools=available_tools,
                            messages=messages
                        )
                    except Exception as api_error:
                        return {"response": f"❌ **API Error: {str(api_error)}**", "tokens_used": 0}

                    if len(response.content) == 1 and response.content[0].type == "text":
                        full_response += response.content[0].text
                        process_query = False
            
            if tools_used:
                tools_info = "\n\n---\n🔧 **Tools Used:** " + ", ".join(set(tools_used))
                full_response += tools_info
            
            return {
                "response": full_response,
                "tokens_used": 0  # No token tracking
            }
            
        except Exception as e:
            self._connection_status = "offline"
            return {"response": f"Error: {str(e)}", "tokens_used": 0}

    async def cleanup(self):
        """Clean up the persistent connection"""
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.exit_stack = None
            self.session = None
            self.stdio = None
            self.write = None
            self._connection_status = "disconnected"


# Create global instance
mcp_interface = MCPInterface()


async def run_mcp_query(query: str, enabled_tools: List[str] = None, model: str = None, session_id: str = "default", websocket = None, send_message_callback=None) -> Dict:
    return await mcp_interface.process_query_with_anthropic(query, enabled_tools, model, session_id, websocket, send_message_callback)
