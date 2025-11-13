"""
MCP Interface for BroadAxis RFP/RFQ Management Platform
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from error_handler import BroadAxisError, ExternalAPIError, error_handler
from token_manager import token_manager, TaskComplexity

logger = logging.getLogger(__name__)

def convert_textblocks_to_string(content):
    """Convert TextBlock objects to JSON-serializable format for Anthropic API"""
    if isinstance(content, list):
        result = []
        for item in content:
            if hasattr(item, 'text'):
                result.append({"type": "text", "text": item.text})
            elif isinstance(item, dict) and 'text' in item:
                result.append({"type": "text", "text": item['text']})
            elif isinstance(item, dict) and 'type' in item:
                result.append(item)  # Already properly formatted
            else:
                result.append({"type": "text", "text": str(item)})
        return result
    elif hasattr(content, 'text'):
        return [{"type": "text", "text": content.text}]
    elif isinstance(content, dict) and 'text' in content:
        return [{"type": "text", "text": content['text']}]
    elif isinstance(content, dict) and 'type' in content:
        return [content]  # Already properly formatted
    else:
        return [{"type": "text", "text": str(content)}]


class MCPInterface:
    def __init__(self):
        # Ensure environment variables are available to the MCP server subprocess
        import os
        from dotenv import load_dotenv
        
        # Load environment variables from parent directory
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
        load_dotenv()  # Also try current directory
        
        # Pass current environment to subprocess
        env = os.environ.copy()
        
        self.server_params = StdioServerParameters(
            command='python',
            args=[os.path.join(os.path.dirname(__file__), "ba-server", "server.py")],
            env=env,
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
            try:
                await self.initialize()
            except Exception as e:
                error_handler.log_error(e, {'operation': 'ensure_connection'})
                self._connection_status = "offline"
                raise ExternalAPIError("Failed to establish MCP connection", {'original_error': str(e)})
    
    async def initialize(self):
        """Initialize connection to MCP server and fetch tools/prompts"""
        async with self._cache_lock:
            if self._tools_cache is None or self._prompts_cache is None or self.session is None:
                self._initializing = True
                try:
                    self._connection_status = "connecting"
                    print(f"🔧 MCP Interface: Starting initialization...")
                    print(f"🔧 Server params: {self.server_params}")
                    
                    # Create persistent connection
                    if self.exit_stack is None:
                        self.exit_stack = AsyncExitStack()
                    
                    print(f"🔧 Creating stdio transport...")
                    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(self.server_params))
                    self.stdio, self.write = stdio_transport
                    
                    print(f"🔧 Creating client session...")
                    self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
                    
                    print(f"🔧 Initializing session...")
                    await self.session.initialize()
                    self._connection_status = "connected"
                    print(f"✅ MCP Interface: Connection established")
                    
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
                    
                    # Fetch full prompt content using get_prompt
                    self._prompts_cache = []
                    for prompt in prompts_response.prompts:
                        try:
                            # Get the full prompt content using get_prompt
                            prompt_content = await self.session.get_prompt(prompt.name, arguments={})
                            print(f"✅ Successfully got prompt content for '{prompt.name}'")
                            
                            # Extract text content from GetPromptResult messages
                            prompt_text = ""
                            if prompt_content and hasattr(prompt_content, 'messages'):
                                for message in prompt_content.messages:
                                    if hasattr(message, 'content'):
                                        # Handle different content types
                                        if isinstance(message.content, list):
                                            for content_item in message.content:
                                                if hasattr(content_item, 'text'):
                                                    prompt_text += content_item.text
                                        elif hasattr(message.content, 'text'):
                                            prompt_text += message.content.text
                                        elif isinstance(message.content, str):
                                            prompt_text += message.content
                            
                            print(f"📄 Extracted prompt text length: {len(prompt_text)} characters")
                            
                            self._prompts_cache.append({
                                "name": prompt.name,
                                "description": prompt.description,
                                "content": prompt_text if prompt_text else prompt.description  # Fallback to description if content is empty
                            })
                        except Exception as e:
                            print(f"⚠️ Failed to get content for prompt {prompt.name}: {e}")
                            # Fallback to just name and description
                            self._prompts_cache.append({
                                "name": prompt.name,
                                "description": prompt.description,
                                "content": prompt.description
                            })
                    
                except Exception as e:
                    self._connection_status = "offline"
                    self._tools_cache = self._tools_cache or []
                    self._prompts_cache = self._prompts_cache or []
                    
                    print(f"❌ MCP Interface: Initialization failed: {e}")
                    print(f"❌ Error type: {type(e)}")
                    import traceback
                    print(f"❌ Traceback: {traceback.format_exc()}")
                    
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
    
    def _determine_task_complexity(self, query: str, enabled_tools: List[str] = None) -> TaskComplexity:
        """Determine task complexity based on query and tools"""
        query_lower = query.lower()
        
        # Complex complexity indicators
        if any(keyword in query_lower for keyword in [
            'analyze', 'comprehensive', 'detailed', 'complete', 'thorough',
            'rfp', 'rfq', 'rfi', 'proposal', 'go/no-go', 'recommendation'
        ]):
            return TaskComplexity.COMPLEX
        
        # Medium complexity indicators
        if any(keyword in query_lower for keyword in [
            'search', 'find', 'list', 'compare', 'evaluate', 'assess'
        ]) or (enabled_tools and len(enabled_tools) > 2):
            return TaskComplexity.MEDIUM
        
        # Default to simple complexity
        return TaskComplexity.SIMPLE
    
    async def process_query_with_anthropic(self, query: str, enabled_tools: List[str] = None, model: str = None, session_id: str = "default", websocket = None, send_message_callback=None, system_prompt_override: str = None) -> Dict:
        if not self.anthropic:
            return {"response": "Anthropic API not available", "tokens_used": 0}
        
        try:
            # Ensure we have an active connection
            await self._ensure_connection()
            
            available_tools = await self.get_tools()
            
            # Filter tools based on query context
            # IMPORTANT: empty list means "no tools allowed"; None means "allow all"
            if enabled_tools is not None:
                available_tools = [tool for tool in available_tools if tool["name"] in enabled_tools]
            
            system_prompt = """I'm BroadAxis AI. I have access to several tools to help you:

1. **Broadaxis_knowledge_search** - Use this tool for ANY questions about BroadAxis company, including:
   - Company information, location, headquarters
   - Company history, background, services
   - Employee information, team details
   - Company policies, procedures
   - Any BroadAxis-specific information

2. **web_search_tool** - Use this for current market research, news, or information not in our knowledge base

3. **Other tools** - For document generation, SharePoint operations, etc.

ALWAYS use the Broadaxis_knowledge_search tool first when asked about BroadAxis company information, location, or any company-specific details.

**IMPORTANT: Be concise and direct. For simple tasks like math calculations, provide the answer without unnecessary explanations.**"""

            # Allow callers to override the system prompt (e.g., trading planner)
            if system_prompt_override:
                system_prompt = system_prompt_override
            
            # Determine task complexity and select appropriate model
            task_complexity = self._determine_task_complexity(query, enabled_tools)
            estimated_tokens = token_manager.estimate_tokens(query + system_prompt)
            
            # Use token manager to select the best model
            if model:
                selected_model = model
            else:
                selected_model = token_manager.get_recommended_model(query, estimated_tokens)
            
            # Generate unique request ID
            import uuid
            request_id = str(uuid.uuid4())
            
            # Track tokens (no restrictions, just monitoring)
            await token_manager.reserve_tokens(selected_model, estimated_tokens, request_id, session_id)
            
            # Build messages for the API call
            messages = [{"role": "user", "content": query}]
            
            # Simple API call without retry logic
            # Set max_tokens based on model capabilities:
            # - Claude 3.5 Sonnet: 8192 tokens max
            # - Claude 3.7 Sonnet: 8192 tokens max  
            # - Claude Opus 4: 16384 tokens max
            # - Claude Haiku: 4096 tokens max
            if "opus" in selected_model.lower() and "4" in selected_model:
                max_output_tokens = 16384  # Opus 4 supports 16K
            elif "sonnet" in selected_model.lower():
                max_output_tokens = 8192   # Sonnet 3.5/3.7 supports 8K
            else:
                max_output_tokens = 4096   # Haiku and others support 4K
            
            try:
                response = await self.anthropic.messages.create(
                    max_tokens=max_output_tokens,
                    model=selected_model,
                    system=system_prompt,
                    tools=available_tools,
                    messages=messages
                )
                
                # Send progress update for initial processing
                if websocket and send_message_callback:
                    try:
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
                    except Exception as ws_error:
                        error_handler.log_error(ws_error, {'operation': 'websocket_processing_update'})
                
            except Exception as api_error:
                # Ensure error message is always a string
                error_msg = str(api_error)
                # If error contains TextBlock objects, convert them
                if "TextBlock" in error_msg:
                    error_msg = f"API Error occurred: {error_msg}"
                return {"response": f"❌ **API Error: {error_msg}**", "tokens_used": 0}
            
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
                        try:
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
                        except Exception as ws_error:
                            error_handler.log_error(ws_error, {'operation': 'websocket_tool_execution_update'})
                    
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
                                    
                                    # Debug: Log the result structure
                                    logger.info(f"Tool {tool_content.name} result type: {type(result)}")
                                    logger.info(f"Tool {tool_content.name} result content type: {type(result.content) if hasattr(result, 'content') else 'No content attr'}")
                                    
                                    # Convert MCP result to proper Anthropic format
                                    content = convert_textblocks_to_string(result.content)
                                    logger.info(f"Converted content type: {type(content)}")
                                    
                                    return {
                                        "type": "tool_result",
                                        "tool_use_id": tool_content.id,
                                        "content": content
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
                            try:
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
                            except Exception as ws_error:
                                error_handler.log_error(ws_error, {'operation': 'websocket_progress_update'})
                        return result
                    
                    tool_results = await asyncio.gather(
                        *[execute_tool_with_progress(tool) for tool in tool_calls],
                        return_exceptions=True
                    )
                    
                    # Add tool results as individual messages
                    for result in tool_results:
                        if isinstance(result, dict):
                            messages.append({"role": "user", "content": [result]})
                        else:
                            error_handler.log_error(result, {'operation': 'parallel_tool_execution'})
                    
                    # --- Simple follow-up call without retry logic ---
                    # Send progress update for response generation
                    if websocket and send_message_callback:
                        try:
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
                        except Exception as ws_error:
                            error_handler.log_error(ws_error, {'operation': 'websocket_generation_update'})

                    try:
                        # Use the same max_tokens logic as the initial call
                        if "opus" in selected_model.lower() and "4" in selected_model:
                            max_output_tokens = 16384  # Opus 4 supports 16K
                        elif "sonnet" in selected_model.lower():
                            max_output_tokens = 8192   # Sonnet 3.5/3.7 supports 8K
                        else:
                            max_output_tokens = 4096   # Haiku and others support 4K
                        
                        response = await self.anthropic.messages.create(
                            max_tokens=max_output_tokens,
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
            
            # Record actual token usage
            actual_input_tokens = estimated_tokens  # Rough estimate
            actual_output_tokens = token_manager.estimate_tokens(full_response)
            total_tokens = actual_input_tokens + actual_output_tokens
            
            token_manager.record_usage(
                selected_model,
                actual_input_tokens,
                actual_output_tokens,
                request_id,
                task_complexity.value,
                session_id
            )
            
            return {
                "response": full_response,
                "tokens_used": total_tokens,
                "input_tokens": actual_input_tokens,
                "output_tokens": actual_output_tokens,
                "model_used": selected_model,
                "request_id": request_id
            }
            
        except Exception as e:
            self._connection_status = "offline"
            # Ensure error message is always a string
            error_msg = str(e)
            # If error contains TextBlock objects, convert them
            if "TextBlock" in error_msg:
                error_msg = f"Processing error occurred: {error_msg}"
            return {"response": f"Error: {error_msg}", "tokens_used": 0}


# Global instance
mcp_interface = MCPInterface()

async def run_mcp_query(
    query: str,
    enabled_tools: List[str] = None,
    model: str = "claude-haiku-4-5-20251001",  # Default to Haiku for cost efficiency
    session_id: str = "default",
    system_prompt: str = None,
    websocket = None,
    send_message_callback = None
) -> Dict:
    """
    Run MCP query with the specified parameters.
    """
    return await mcp_interface.process_query_with_anthropic(
        query=query,
        enabled_tools=enabled_tools,
        model=model,
        session_id=session_id,
        websocket=websocket,
        send_message_callback=send_message_callback,
        system_prompt_override=system_prompt
    )
