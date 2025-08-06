"""
FastAPI Backend for BroadAxis RFP/RFQ Management Platform
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict

import nest_asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ValidationError as PydanticValidationError

from error_handler import (
    BroadAxisError, ValidationError, AuthenticationError, NotFoundError,
    ExternalAPIError, FileOperationError, error_handler
)

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("Warning: tiktoken not available - token counting disabled")

nest_asyncio.apply()

app = FastAPI(title="BroadAxis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handlers
@app.exception_handler(BroadAxisError)
async def broadaxis_exception_handler(request: Request, exc: BroadAxisError):
    error_info = error_handler.log_error(exc, {'endpoint': str(request.url)})
    return JSONResponse(
        status_code=exc.status_code,
        content=error_handler.format_error_response(exc)
    )

@app.exception_handler(PydanticValidationError)
async def validation_exception_handler(request: Request, exc: PydanticValidationError):
    validation_error = ValidationError(
        "Request validation failed",
        details={'validation_errors': exc.errors()}
    )
    error_info = error_handler.log_error(validation_error, {'endpoint': str(request.url)})
    return JSONResponse(
        status_code=400,
        content=error_handler.format_error_response(validation_error)
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    error_info = error_handler.log_error(exc, {'endpoint': str(request.url)})
    return JSONResponse(
        status_code=500,
        content=error_handler.format_error_response(exc)
    )



class ChatRequest(BaseModel):
    query: str
    enabled_tools: List[str] = []
    model: str = "claude-3-7-sonnet-20250219"

class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    tokens_used: int = 0
    tokens_remaining: int = 0

class TokenManager:
    def __init__(self):
        self.MAX_TOKENS_PER_REQUEST = 8000
        self.MAX_TOKENS_PER_SESSION = 50000
        self.MAX_TOKENS_PER_DAY = 200000
        self.session_usage = defaultdict(int)
        self.daily_usage = defaultdict(int)
        self.daily_reset = defaultdict(lambda: datetime.now().date())
        self.encoding = tiktoken.get_encoding("cl100k_base") if tiktoken else None
    
    def count_tokens(self, text: str) -> int:
        try:
            if not text:
                return 0
            if not self.encoding:
                return len(text) // 4  # Rough estimate
            return len(self.encoding.encode(text))
        except Exception as e:
            error_handler.log_error(e, {'text_length': len(text) if text else 0})
            return len(text) // 4 if text else 0  # Fallback estimate
    
    def count_messages_tokens(self, messages: List[Dict], system_prompt: str = "") -> int:
        total = self.count_tokens(system_prompt)
        for msg in messages:
            if isinstance(msg.get('content'), str):
                total += self.count_tokens(msg['content'])
            elif isinstance(msg.get('content'), list):
                for content in msg['content']:
                    if isinstance(content, dict) and 'text' in content:
                        total += self.count_tokens(content['text'])
        return total
    
    def check_limits(self, session_id: str, estimated_tokens: int) -> Dict:
        try:
            # Reset daily usage if needed
            today = datetime.now().date()
            if self.daily_reset[session_id] < today:
                self.daily_usage[session_id] = 0
                self.daily_reset[session_id] = today
            
            current_session = self.session_usage[session_id]
            current_daily = self.daily_usage[session_id]
            
            if estimated_tokens > self.MAX_TOKENS_PER_REQUEST:
                return {"allowed": False, "reason": "Request exceeds per-request limit"}
            
            if current_session + estimated_tokens > self.MAX_TOKENS_PER_SESSION:
                return {"allowed": False, "reason": "Session limit exceeded"}
            
            if current_daily + estimated_tokens > self.MAX_TOKENS_PER_DAY:
                return {"allowed": False, "reason": "Daily limit exceeded"}
            
            return {
                "allowed": True,
                "session_remaining": self.MAX_TOKENS_PER_SESSION - current_session,
                "daily_remaining": self.MAX_TOKENS_PER_DAY - current_daily
            }
        except Exception as e:
            error_handler.log_error(e, {'session_id': session_id, 'estimated_tokens': estimated_tokens})
            return {"allowed": False, "reason": "Token limit check failed"}
    
    def add_usage(self, session_id: str, tokens_used: int):
        self.session_usage[session_id] += tokens_used
        self.daily_usage[session_id] += tokens_used
    
    def get_usage(self, session_id: str) -> Dict:
        today = datetime.now().date()
        if self.daily_reset[session_id] < today:
            self.daily_usage[session_id] = 0
            self.daily_reset[session_id] = today
        
        return {
            "session_used": self.session_usage[session_id],
            "session_limit": self.MAX_TOKENS_PER_SESSION,
            "daily_used": self.daily_usage[session_id],
            "daily_limit": self.MAX_TOKENS_PER_DAY,
            "request_limit": self.MAX_TOKENS_PER_REQUEST
        }

token_manager = TokenManager()



class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            error_handler.log_error(e, {'operation': 'websocket_send_message'})
            self.disconnect(websocket)
            raise

manager = ConnectionManager()

class MCPInterface:
    def __init__(self):
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.join(os.path.dirname(__file__), "..", "ba-server", "server.py")],
            env=None,
        )
        self.anthropic = None
        self._tools_cache = None
        self._prompts_cache = None
        self._cache_lock = asyncio.Lock()
        self._connection_status = "disconnected"
        self._tools_loading = False
        self._prompts_loading = False
        try:
            from anthropic import Anthropic
            self.anthropic = Anthropic()
        except ImportError:
            print("Warning: Anthropic not available")
    
    async def get_tools(self):
        """Get cached tools or fetch from server"""
        async with self._cache_lock:
            if self._tools_cache is None:
                self._tools_loading = True
                try:
                    self._connection_status = "connecting"
                    async with stdio_client(self.server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            self._connection_status = "connected"
                            tools_response = await session.list_tools()
                            
                            if not tools_response or not hasattr(tools_response, 'tools'):
                                raise ExternalAPIError("Invalid tools response from MCP server")
                            
                            self._tools_cache = [{
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema
                            } for tool in tools_response.tools]
                            
                except Exception as e:
                    self._connection_status = "offline"
                    self._tools_cache = []
                    
                    if isinstance(e, BroadAxisError):
                        raise e
                    else:
                        error_handler.log_error(e, {'operation': 'get_tools'})
                        raise ExternalAPIError("Failed to fetch tools from MCP server", 
                                             {'original_error': str(e)})
                finally:
                    self._tools_loading = False
            return self._tools_cache
    
    async def get_prompts(self):
        """Get cached prompts or fetch from server"""
        async with self._cache_lock:
            if self._prompts_cache is None:
                self._prompts_loading = True
                try:
                    self._connection_status = "connecting"
                    async with stdio_client(self.server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            self._connection_status = "connected"
                            prompts_response = await session.list_prompts()
                            
                            if not prompts_response or not hasattr(prompts_response, 'prompts'):
                                raise ExternalAPIError("Invalid prompts response from MCP server")
                            
                            self._prompts_cache = [{
                                "name": prompt.name,
                                "description": prompt.description
                            } for prompt in prompts_response.prompts]
                            
                except Exception as e:
                    self._connection_status = "offline"
                    self._prompts_cache = []
                    
                    if isinstance(e, BroadAxisError):
                        raise e
                    else:
                        error_handler.log_error(e, {'operation': 'get_prompts'})
                        raise ExternalAPIError("Failed to fetch prompts from MCP server", 
                                             {'original_error': str(e)})
                finally:
                    self._prompts_loading = False
            return self._prompts_cache
    
    async def process_query_with_anthropic(self, query: str, enabled_tools: List[str] = None, model: str = None, session_id: str = "default") -> Dict:
        if not self.anthropic:
            return {"response": "Anthropic API not available", "tokens_used": 0}
            
        try:
            # Handle MCP prompt invocations
            if query.startswith('PROMPT:'):
                prompt_name = query[7:]
                try:
                    async with stdio_client(self.server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            prompts = await self.get_prompts()
                            target_prompt = next((p for p in prompts if p["name"] == prompt_name), None)
                            
                            if target_prompt:
                                prompt_result = await session.get_prompt(prompt_name, arguments={})
                                prompt_content = prompt_result.messages[0].content.text if prompt_result.messages else ""
                                enhanced_system = prompt_content + "\n\nIMPORTANT: Format your response with clear headings, bullet points, and proper spacing for readability."
                                
                                selected_model = model or "claude-3-7-sonnet-20250219"
                                
                                # Token management for prompts
                                estimated_tokens = token_manager.count_tokens(enhanced_system + "Analyze the uploaded documents using this framework.")
                                limit_check = token_manager.check_limits(session_id, estimated_tokens)
                                
                                if not limit_check["allowed"]:
                                    return {"response": f"Token limit exceeded: {limit_check['reason']}", "tokens_used": 0}
                                
                                response = self.anthropic.messages.create(
                                    max_tokens=min(3048, limit_check.get("session_remaining", 3048)),
                                    model=selected_model,
                                    system=enhanced_system,
                                    messages=[{'role': 'user', 'content': "Analyze the uploaded documents using this framework."}]
                                )
                                
                                tokens_used = response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else estimated_tokens
                                token_manager.add_usage(session_id, tokens_used)
                                
                                return {
                                    "response": response.content[0].text if response.content else "No response generated",
                                    "tokens_used": tokens_used
                                }
                            else:
                                return {"response": f"Prompt '{prompt_name}' not found", "tokens_used": 0}
                except Exception as e:
                    return {"response": f"Error using prompt: {str(e)}", "tokens_used": 0}
            
            # Use fresh connection for tool-based queries with cached tools
            self._connection_status = "connecting"
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._connection_status = "connected"
                    
                    available_tools = await self.get_tools()
                    
                    if enabled_tools:
                        available_tools = [tool for tool in available_tools if tool["name"] in enabled_tools]
                    
                    system_prompt = """You are BroadAxis-AI, an intelligent RFP/RFQ management assistant. 
Use Broadaxis_knowledge_search for company information first, then web_search_tool for external data if needed.
For uploaded RFP documents, provide analysis and offer Go/No-Go recommendations."""
                    
                    selected_model = model or "claude-3-7-sonnet-20250219"
                    messages = [{'role': 'user', 'content': query}]
                    
                    # Token management
                    estimated_tokens = token_manager.count_messages_tokens(messages, system_prompt)
                    limit_check = token_manager.check_limits(session_id, estimated_tokens)
                    
                    if not limit_check["allowed"]:
                        return {"response": f"Token limit exceeded: {limit_check['reason']}", "tokens_used": 0}
                    
                    max_tokens = min(4024, limit_check.get("session_remaining", 4024))
                    response = self.anthropic.messages.create(
                        max_tokens=max_tokens,
                        model=selected_model,
                        system=system_prompt,
                        tools=available_tools,
                        messages=messages
                    )
                    
                    full_response = ""
                    tools_used = []
                    process_query = True
                    total_tokens_used = 0
                    
                    while process_query:
                        assistant_content = []
                        for content in response.content:
                            if content.type == 'text':
                                full_response += content.text
                                assistant_content.append(content)
                                if len(response.content) == 1:
                                    process_query = False
                            elif content.type == 'tool_use':
                                tools_used.append(content.name)
                                assistant_content.append(content)
                                messages.append({'role': 'assistant', 'content': assistant_content})
                                
                                try:
                                    result = await session.call_tool(content.name, arguments=content.input)
                                    messages.append({
                                        "role": "user",
                                        "content": [{
                                            "type": "tool_result",
                                            "tool_use_id": content.id,
                                            "content": result.content
                                        }]
                                    })
                                except Exception as tool_error:
                                    messages.append({
                                        "role": "user",
                                        "content": [{
                                            "type": "tool_result",
                                            "tool_use_id": content.id,
                                            "content": [{"type": "text", "text": f"Tool failed: {str(tool_error)}"}]
                                        }]
                                    })
                                
                                # Check tokens before follow-up call
                                current_tokens = token_manager.count_messages_tokens(messages, system_prompt)
                                follow_up_check = token_manager.check_limits(session_id, current_tokens)
                                
                                if not follow_up_check["allowed"]:
                                    full_response += "\n\n[Response truncated due to token limits]"
                                    process_query = False
                                    continue
                                
                                response = self.anthropic.messages.create(
                                    max_tokens=min(2024, follow_up_check.get("session_remaining", 2024)),
                                    model=selected_model,
                                    system=system_prompt,
                                    tools=available_tools,
                                    messages=messages
                                )
                                
                                if len(response.content) == 1 and response.content[0].type == "text":
                                    full_response += response.content[0].text
                                    process_query = False
                    
                    # Calculate total tokens used
                    if hasattr(response, 'usage'):
                        total_tokens_used += response.usage.input_tokens + response.usage.output_tokens
                    
                    token_manager.add_usage(session_id, total_tokens_used)
                    
                    if tools_used:
                        tools_info = "\n\n---\n🔧 **Tools Used:** " + ", ".join(set(tools_used))
                        full_response += tools_info
                    
                    return {
                        "response": full_response,
                        "tokens_used": total_tokens_used
                    }
                    
        except Exception as e:
            self._connection_status = "offline"
            return {"response": f"Error: {str(e)}", "tokens_used": 0}

mcp_interface = MCPInterface()

async def run_mcp_query(query: str, enabled_tools: List[str] = None, model: str = None, session_id: str = "default") -> Dict:
    return await mcp_interface.process_query_with_anthropic(query, enabled_tools, model, session_id)

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "BroadAxis FastAPI Backend",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy", 
            "timestamp": time.time(),
            "message": "BroadAxis API is running",
            "version": "1.0.0",
            "mcp_connection": mcp_interface._connection_status,
            "error_logging": "enabled"
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'health_check'})
        return {
            "status": "degraded",
            "timestamp": time.time(),
            "message": "BroadAxis API is running with issues",
            "version": "1.0.0",
            "error": str(e)
        }

@app.get("/api/status")
def get_status():
    """Get current connection and loading status."""
    return {
        "connection_status": mcp_interface._connection_status,
        "tools_loading": mcp_interface._tools_loading,
        "prompts_loading": mcp_interface._prompts_loading,
        "tools_cached": mcp_interface._tools_cache is not None,
        "prompts_cached": mcp_interface._prompts_cache is not None
    }

@app.get("/api/tokens/{session_id}")
def get_token_usage(session_id: str):
    """Get token usage for a specific session."""
    usage = token_manager.get_usage(session_id)
    return {
        "usage": usage,
        "status": "success"
    }

@app.get("/api/tokens")
def get_token_limits():
    """Get token limits and general statistics."""
    return {
        "limits": {
            "per_request": token_manager.MAX_TOKENS_PER_REQUEST,
            "per_session": token_manager.MAX_TOKENS_PER_SESSION,
            "per_day": token_manager.MAX_TOKENS_PER_DAY
        },
        "active_sessions": len(token_manager.session_usage),
        "tiktoken_available": tiktoken is not None,
        "status": "success"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process chat queries through the MCP server."""
    if not request.query or not request.query.strip():
        raise ValidationError("Query cannot be empty")
    
    try:
        session_id = f"api_{int(time.time())}_{hash(request.query) % 10000}"
        result = await run_mcp_query(
            query=request.query.strip(),
            enabled_tools=request.enabled_tools,
            model=request.model,
            session_id=session_id
        )
        
        if not isinstance(result, dict) or "response" not in result:
            raise ExternalAPIError("Invalid response from MCP query processor")
        
        usage = token_manager.get_usage(session_id)
        return ChatResponse(
            response=result["response"],
            status="success",
            tokens_used=result.get("tokens_used", 0),
            tokens_remaining=usage["session_limit"] - usage["session_used"]
        )
    except BroadAxisError:
        raise
    except Exception as e:
        error_handler.log_error(e, {'query_length': len(request.query), 'enabled_tools': request.enabled_tools})
        raise ExternalAPIError("Failed to process chat request", {'original_error': str(e)})

# Store uploaded files per session
session_files = {}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise ValidationError("No file provided")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.txt', '.md', '.docx', '.doc'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_extensions:
        raise ValidationError(f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}")
    
    try:
        file_content = await file.read()
        
        if not file_content:
            raise ValidationError("Uploaded file is empty")
        
        if len(file_content) > 50 * 1024 * 1024:  # 50MB limit
            raise ValidationError("File size exceeds 50MB limit")
        
        # Store file content for chat analysis
        text_content = ""
        if file.filename.lower().endswith('.pdf'):
            try:
                import PyPDF2
                from io import BytesIO
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                text_content = "\n".join([page.extract_text() for page in pdf_reader.pages])
                
                if not text_content.strip():
                    raise FileOperationError("PDF appears to be empty or contains no extractable text")
                    
            except Exception as e:
                error_handler.log_error(e, {'filename': file.filename, 'operation': 'pdf_extraction'})
                raise FileOperationError("Failed to extract text from PDF", {'filename': file.filename})
        else:
            try:
                text_content = file_content.decode('utf-8', errors='ignore')
            except Exception as e:
                error_handler.log_error(e, {'filename': file.filename, 'operation': 'text_decode'})
                raise FileOperationError("Failed to decode text file", {'filename': file.filename})
        
        return {
            "status": "success",
            "filename": file.filename,
            "size": len(file_content),
            "message": f"File '{file.filename}' uploaded and analyzed. You can now ask questions about it.",
            "analysis": f"Document '{file.filename}' uploaded successfully. You can now ask questions about it."
        }
        
    except BroadAxisError:
        raise
    except Exception as e:
        error_handler.log_error(e, {'filename': file.filename, 'operation': 'file_upload'})
        raise FileOperationError("Failed to process uploaded file", {'filename': file.filename})



@app.get("/api/tools")
async def get_available_tools():
    try:
        tools = await mcp_interface.get_tools()
        return {
            "tools": tools, 
            "status": "success",
            "connection_status": mcp_interface._connection_status,
            "loading": mcp_interface._tools_loading
        }
    except BroadAxisError:
        raise
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_available_tools'})
        # Return partial success for UI functionality
        return {
            "tools": [], 
            "status": "error",
            "connection_status": "offline",
            "loading": False,
            "error_message": "Failed to fetch tools from server"
        }

@app.get("/api/prompts")
async def get_available_prompts():
    try:
        prompts = await mcp_interface.get_prompts()
        return {
            "prompts": prompts, 
            "status": "success",
            "connection_status": mcp_interface._connection_status,
            "loading": mcp_interface._prompts_loading
        }
    except Exception as e:
        print(f"Error fetching prompts: {e}")
        return {
            "prompts": [], 
            "status": "error",
            "connection_status": "offline",
            "loading": False
        }





# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication."""
    session_id = f"ws_{id(websocket)}_{int(time.time())}"
    
    try:
        await manager.connect(websocket)
        
        while True:
            try:
                # Receive message from client with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)  # 5 min timeout
                
                try:
                    message_data = json.loads(data)
                except json.JSONDecodeError as e:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format in message",
                            "status": "error"
                        }),
                        websocket
                    )
                    continue

                query = message_data.get("query", "").strip()
                enabled_tools = message_data.get("enabled_tools", [])
                model = message_data.get("model", "claude-3-7-sonnet-20250219")
                
                if not query:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": "Query cannot be empty",
                            "status": "error"
                        }),
                        websocket
                    )
                    continue
                
                # Handle simple responses naturally
                simple_queries = {
                    'hello': "Hello! How can I help you today?",
                    'hi': "Hi there! What can I assist you with?",
                    'hey': "Hey! How can I help?",
                    'test': "I'm working perfectly! How can I assist you?",
                    'awesome': "Glad you think so! What would you like to work on?",
                    'great': "Thanks! How can I help you today?",
                    'good': "Great to hear! What can I do for you?",
                    'thanks': "You're welcome! Anything else I can help with?",
                    'thank you': "You're welcome! Let me know if you need anything else."
                }
                
                query_lower = query.lower().strip()
                if query_lower in simple_queries:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "response",
                            "message": simple_queries[query_lower],
                            "status": "success"
                        }),
                        websocket
                    )
                    continue

                # Send acknowledgment
                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Processing your request..."}),
                    websocket
                )

                try:
                    # Process query with enabled tools
                    result = await run_mcp_query(query, enabled_tools, model, session_id)
                    
                    if not isinstance(result, dict) or "response" not in result:
                        raise ExternalAPIError("Invalid response from query processor")
                    
                    usage = token_manager.get_usage(session_id)

                    # Send response
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "response",
                            "message": result["response"],
                            "status": "success",
                            "tokens_used": result.get("tokens_used", 0),
                            "tokens_remaining": usage["session_limit"] - usage["session_used"],
                            "usage": usage
                        }),
                        websocket
                    )

                except BroadAxisError as e:
                    error_handler.log_error(e, {'session_id': session_id, 'query_length': len(query)})
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": e.message,
                            "error_type": e.error_type.value,
                            "status": "error"
                        }),
                        websocket
                    )
                except Exception as e:
                    error_handler.log_error(e, {'session_id': session_id, 'query_length': len(query)})
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": "An unexpected error occurred while processing your request",
                            "status": "error"
                        }),
                        websocket
                    )
                    
            except asyncio.TimeoutError:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": "Connection timeout - please reconnect",
                        "status": "timeout"
                    }),
                    websocket
                )
                break
                
            except Exception as e:
                error_handler.log_error(e, {'session_id': session_id, 'operation': 'websocket_message_handling'})
                break

    except WebSocketDisconnect:
        error_handler.logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        error_handler.log_error(e, {'session_id': session_id, 'operation': 'websocket_connection'})
    finally:
        manager.disconnect(websocket)

@app.get("/api/files")
async def list_files():
    try:
        files_dir = os.path.join(os.path.dirname(__file__), "..", "ba-server", "generated_files")
        
        if not os.path.exists(files_dir):
            # Create directory if it doesn't exist
            try:
                os.makedirs(files_dir, exist_ok=True)
            except Exception as e:
                error_handler.log_error(e, {'operation': 'create_files_dir', 'path': files_dir})
                raise FileOperationError("Failed to access files directory")
        
        files = []
        try:
            for filename in os.listdir(files_dir):
                file_path = os.path.join(files_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        file_size = os.path.getsize(file_path)
                        file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                        file_type = filename.split('.')[-1] if '.' in filename else "unknown"
                        
                        files.append({
                            "filename": filename,
                            "file_size": file_size,
                            "modified_at": file_modified.isoformat(),
                            "type": file_type
                        })
                    except Exception as e:
                        error_handler.log_error(e, {'operation': 'file_stat', 'filename': filename})
                        # Skip problematic files but continue listing
                        continue
        except Exception as e:
            error_handler.log_error(e, {'operation': 'list_directory', 'path': files_dir})
            raise FileOperationError("Failed to list files in directory")
        
        return {"files": files, "status": "success"}
        
    except BroadAxisError:
        raise
    except Exception as e:
        error_handler.log_error(e, {'operation': 'list_files'})
        raise FileOperationError("Failed to retrieve file list")

@app.get("/api/files/{filename}")
async def download_file(filename: str):
    if not filename or '..' in filename or '/' in filename or '\\' in filename:
        raise ValidationError("Invalid filename")
    
    try:
        files_dir = os.path.join(os.path.dirname(__file__), "..", "ba-server", "generated_files")
        file_path = os.path.join(files_dir, filename)
        
        # Security check - ensure file is within the allowed directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(files_dir)):
            raise ValidationError("Access denied: Invalid file path")
        
        if not os.path.exists(file_path):
            raise NotFoundError(f"File '{filename}' not found")
        
        if not os.path.isfile(file_path):
            raise ValidationError(f"'{filename}' is not a file")
        
        try:
            return FileResponse(
                path=file_path,
                filename=filename,
                media_type='application/octet-stream'
            )
        except Exception as e:
            error_handler.log_error(e, {'operation': 'file_response', 'filename': filename})
            raise FileOperationError(f"Failed to serve file '{filename}'")
            
    except BroadAxisError:
        raise
    except Exception as e:
        error_handler.log_error(e, {'operation': 'download_file', 'filename': filename})
        raise FileOperationError(f"Failed to download file '{filename}'")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)