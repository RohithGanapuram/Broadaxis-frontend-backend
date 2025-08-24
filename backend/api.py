"""
FastAPI Backend for BroadAxis RFP/RFQ Management Platform
"""

import asyncio
import json
import os
import sys
import time
import random
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import imaplib
import requests
import email
from urllib.parse import urlparse
from pathlib import Path
import re


import nest_asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ValidationError as PydanticValidationError
from dotenv import load_dotenv
from limiter import MinuteBudgetLimiter

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

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



class EmailFetchRequest(BaseModel):
    email_accounts: List[str] = []  # Optional: specific accounts to fetch from
    use_real_email: bool = True  # Default to real email (Graph API)
    use_graph_api: bool = True  # Default to Microsoft Graph API
 
 
class EmailAttachment(BaseModel):
    filename: str
    file_path: Optional[str] = None  # Optional for link attachments
    file_size: Optional[int] = None  # Optional for link attachments
    download_date: str
    type: str = "file"  # "file" or "link"
    url: Optional[str] = None  # For link attachments
    domain: Optional[str] = None  # For link attachments
 
class FetchedEmail(BaseModel):
    email_id: str
    sender: str
    subject: str
    date: str
    account: str
    attachments: List[EmailAttachment]
    has_rfp_keywords: bool
 
class EmailFetchResponse(BaseModel):
    status: str
    message: str
    emails_found: int
    attachments_downloaded: int
    fetched_emails: List[FetchedEmail]

class TokenManager:
    def __init__(self):
        self.MAX_TOKENS_PER_REQUEST = 50000
        self.MAX_TOKENS_PER_SESSION = 200000
        self.MAX_TOKENS_PER_DAY = 300000
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
        
        # legacy fixed-interval throttle (unused after limiter below)
        self._last_request_time = 0
        self._min_request_interval = 3.0  # Minimum 3 seconds between requests


        try:
            from anthropic import AsyncAnthropic
            self.anthropic = AsyncAnthropic()
        except ImportError:
            print("Warning: Anthropic not available")

        # NEW: server-wide Anthropic budget limiter
        self._budget = MinuteBudgetLimiter(
            rpm=int(os.getenv("ANTHROPIC_RPM", 60)),
            max_in_flight=int(os.getenv("ANTHROPIC_MAX_IN_FLIGHT", 6)),
        )
        
        # Per-session throttling
        self._last_by_session = defaultdict(float)
        self._min_request_interval = float(os.getenv("ANTHROPIC_MIN_INTERVAL_SEC", 0.5))
        
        # Model-specific token limits and configurations
        self._model_configs = {
            "claude-3-7-sonnet-20250219": {
                "max_output_tokens": 8192,  # Claude 3.7 Sonnet limit
                "recommended_first_call": 4096,
                "recommended_follow_up": 2048,
                "description": "Claude 3.7 Sonnet"
            },
            "claude-sonnet-4-20250514": {
                "max_output_tokens": 8192,  # Claude 4 Sonnet limit
                "recommended_first_call": 4096,
                "recommended_follow_up": 2048,
                "description": "Claude 4 Sonnet"
            },
            "claude-opus-4-20250514": {
                "max_output_tokens": 8192,  # Claude 4 Opus limit
                "recommended_first_call": 4096,
                "recommended_follow_up": 2048,
                "description": "Claude 4 Opus"
            }
        }
        

    def _retry_after_secs(self, e, fallback: float) -> float:
        try:
            # Anthropic SDK exceptions typically have a .response with headers
            headers = getattr(getattr(e, "response", None), "headers", {}) or {}
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra:
                return float(ra)
        except Exception:
            pass
        return fallback

    def _is_rate_limited(self, e) -> bool:
        status = getattr(e, "status_code", None)
        return status == 429 or "Too Many Requests" in str(e)
    
    def _calculate_max_tokens(self, model: str, session_remaining: int, call_type: str = "first") -> int:
        """
        Calculate adaptive max_tokens based on model limits and user allowance.
        
        Args:
            model: The model name
            session_remaining: Remaining tokens in the session
            call_type: "first" for initial call, "follow_up" for subsequent calls
            
        Returns:
            Adaptive max_tokens value
        """
        # Get model configuration, fallback to Claude 3.7 Sonnet defaults
        model_config = self._model_configs.get(model, self._model_configs["claude-3-7-sonnet-20250219"])
        
        # Get recommended limits for this call type
        if call_type == "first":
            recommended = model_config["recommended_first_call"]
        else:  # follow_up
            recommended = model_config["recommended_follow_up"]
        
        # Calculate adaptive limit: min(user_allowance, model_limit, recommended)
        model_limit = model_config["max_output_tokens"]
        
        # Use the most restrictive of the three limits
        adaptive_limit = min(session_remaining, model_limit, recommended)
        
        # Ensure we don't go below a reasonable minimum
        min_limit = 512 if call_type == "first" else 256
        final_limit = max(adaptive_limit, min_limit)
        
        # Log the calculation for debugging (only in development)
        if os.getenv("DEBUG_TOKEN_LIMITS", "false").lower() == "true":
            print(f"Token calculation for {model} ({call_type}): "
                  f"session_remaining={session_remaining}, "
                  f"model_limit={model_limit}, "
                  f"recommended={recommended}, "
                  f"final={final_limit}")
        
        return final_limit
    
    def get_model_info(self, model: str) -> dict:
        """
        Get information about a specific model's configuration.
        
        Args:
            model: The model name
            
        Returns:
            Dictionary with model configuration information
        """
        model_config = self._model_configs.get(model, self._model_configs["claude-3-7-sonnet-20250219"])
        return {
            "model": model,
            "description": model_config["description"],
            "max_output_tokens": model_config["max_output_tokens"],
            "recommended_first_call": model_config["recommended_first_call"],
            "recommended_follow_up": model_config["recommended_follow_up"]
        }
    
    async def initialize(self):
        """Initialize both tools and prompts in a single connection"""
        async with self._cache_lock:
            if self._tools_cache is None or self._prompts_cache is None:
                self._initializing = True
                try:
                    self._connection_status = "connecting"
                    async with stdio_client(self.server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            self._connection_status = "connected"
                            
                            # Fetch both tools and prompts in parallel
                            tools_task = session.list_tools()
                            prompts_task = session.list_prompts()
                            
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
    
    async def process_query_with_anthropic(self, query: str, enabled_tools: List[str] = None, model: str = None, session_id: str = "default", websocket = None) -> Dict:
        if not self.anthropic:
            return {"response": "Anthropic API not available", "tokens_used": 0}
        
        # Initialize token tracking
        total_in = total_out = 0
        
        try:
            # Use fresh connection for tool-based queries with cached tools
            self._connection_status = "connecting"
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._connection_status = "connected"
                    
                    available_tools = await self.get_tools()
                    
                    # Smart tool selection based on query type
                    needs_broadaxis_search = any(keyword in query.lower() for keyword in [
                        'broadaxis', 'company', 'our', 'we', 'us', 'experience', 'capability', 'past project', 
                        'rfp', 'rfi', 'rfq', 'proposal', 'bid', 'contract', 'client', 'portfolio'
                    ])
                    
                    needs_web_search = any(keyword in query.lower() for keyword in [
                        'market', 'industry', 'competitor', 'trend', 'research', 'latest', 'current', 
                        'news', 'technology', 'standard', 'regulation', 'compliance'
                    ])
                    
                    # Filter tools based on query context
                    if enabled_tools:
                        available_tools = [tool for tool in available_tools if tool["name"] in enabled_tools]
                    elif needs_broadaxis_search and not needs_web_search:
                        available_tools = [tool for tool in available_tools if tool["name"] == "Broadaxis_knowledge_search"]
                    elif needs_web_search and not needs_broadaxis_search:
                        available_tools = [tool for tool in available_tools if tool["name"] == "web_search_tool"]
                    elif not needs_broadaxis_search and not needs_web_search:
                        # Basic conversation - no tools needed
                        available_tools = []
                    
                    system_prompt = "I'm BroadAxis AI. For company questions, I search our knowledge base first. For market research, I use web search. I provide direct answers for general conversation. IMPORTANT: If multiple tools look useful, choose the single highest-value tool first, then wait for results before deciding on another. This ensures efficient processing and prevents overwhelming the system."
                    
                    selected_model = model or "claude-3-7-sonnet-20250219"
                    messages = [{'role': 'user', 'content': query}]
                    
                    # Token management
                    estimated_tokens = token_manager.count_messages_tokens(messages, system_prompt)
                    limit_check = token_manager.check_limits(session_id, estimated_tokens)
                    
                    if not limit_check["allowed"]:
                        return {"response": f"Token limit exceeded: {limit_check['reason']}", "tokens_used": 0}
                    
                    max_tokens = self._calculate_max_tokens(selected_model, limit_check.get("session_remaining", 4096), "first")
                    
                    # --- Minute-budgeted Anthropic call (FIRST call) ---
                    max_retries = 2
                    base_delay = 2.0
                    
                    # Per-session throttling
                    now = time.time()
                    since = now - self._last_by_session.get(session_id, 0)
                    if since < self._min_request_interval:
                        await asyncio.sleep(self._min_request_interval - since)

                    for attempt in range(max_retries + 1):
                        # Budget reservation for each attempt
                        await self._budget.reserve()
                        try:
                            response = await self.anthropic.messages.create(
                                max_tokens=max_tokens,
                                model=selected_model,
                                system=system_prompt,
                                tools=available_tools,
                                messages=messages
                            )

                            # usage tally
                            if hasattr(response, "usage"):
                                total_in += getattr(response.usage, "input_tokens", 0)
                                total_out += getattr(response.usage, "output_tokens", 0)
                            
                            # Send progress update for initial processing
                            if websocket:
                                await manager.send_personal_message(
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
                            
                            break  # Success, exit retry loop
                            
                        except Exception as api_error:
                            if self._is_rate_limited(api_error):
                                delay = self._retry_after_secs(api_error, base_delay * (3 ** attempt) + random.uniform(1,3))
                                if websocket:
                                    await manager.send_personal_message(
                                        json.dumps({
                                            "type": "progress",
                                            "message": f"Rate limit hit, waiting {delay:.1f}s before retry... (Attempt {attempt + 1}/{max_retries + 1})",
                                            "progress": 20 + (attempt + 1) / (max_retries + 1) * 30,
                                            "step": "rate_limit_retry",
                                            "current_step": attempt + 1,
                                            "total_steps": max_retries + 1
                                        }), 
                                        websocket
                                    )
                                await asyncio.sleep(delay)
                                continue
                            elif "529" in str(api_error) or "Overloaded" in str(api_error) or "overloaded_error" in str(api_error):
                                if attempt < max_retries:
                                    delay = self._retry_after_secs(api_error, base_delay * (4 ** attempt) + random.uniform(2, 5))
                                    if websocket:
                                        await manager.send_personal_message(
                                            json.dumps({
                                                "type": "progress",
                                                "message": f"Anthropic servers overloaded, waiting {delay:.1f}s before retry... (Attempt {attempt + 1}/{max_retries + 1})",
                                                "progress": 20 + (attempt + 1) / (max_retries + 1) * 30,
                                                "step": "server_overload_retry",
                                                "current_step": attempt + 1,
                                                "total_steps": max_retries + 1
                                            }),
                                            websocket
                                        )
                                    await asyncio.sleep(delay)
                                    continue
                                else:
                                    return {"response": "⚠️ **Anthropic servers are currently overloaded. Please wait 3-5 minutes before trying again, or try a simpler query.**", "tokens_used": 0}
                            else:
                                return {"response": f"❌ **API Error: {str(api_error)}**", "tokens_used": 0}
                        finally:
                            self._budget.release()
                    
                    # Update session timestamp only after successful call
                    self._last_by_session[session_id] = time.time()
                    
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
                            if websocket:
                                tool_names = [tool.name for tool in tool_calls]
                                progress_message = f"Executing {len(tool_calls)} tools: {', '.join(tool_names[:2])}"
                                if len(tool_names) > 2:
                                    progress_message += f" and {len(tool_names) - 2} more..."
                                
                                await manager.send_personal_message(
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
                            
                            # Adaptive concurrency based on remaining tokens and model
                            remaining = limit_check.get("session_remaining", 4096)
                            model_config = self._model_configs.get(selected_model, self._model_configs["claude-3-7-sonnet-20250219"])
                            
                            # More conservative limits for models with higher token limits
                            if model_config["max_output_tokens"] > 8000:
                                TOOL_CONCURRENCY = 1 if remaining < 2000 else min(int(os.getenv("MCP_TOOL_CONCURRENCY", 1)), 2)
                                MAX_TOOL_CALLS_PER_TURN = 1 if remaining < 2500 else 2
                            else:
                                TOOL_CONCURRENCY = 1 if remaining < 1500 else min(int(os.getenv("MCP_TOOL_CONCURRENCY", 1)), 2)
                                MAX_TOOL_CALLS_PER_TURN = 1 if remaining < 2000 else 2
                            
                            # Adaptive tool result size based on remaining tokens and model
                            model_config = self._model_configs.get(selected_model, self._model_configs["claude-3-7-sonnet-20250219"])
                            if remaining < 1000:
                                tool_result_max_chars = 2000
                            elif remaining < 2000:
                                tool_result_max_chars = 3000
                            else:
                                # Scale based on model's max output tokens
                                tool_result_max_chars = min(4000, model_config["max_output_tokens"] // 2)
                            
                            # Apply the adaptive cap to tool calls
                            if len(tool_calls) > MAX_TOOL_CALLS_PER_TURN:
                                # Keep only the first tool call and notify the user
                                tool_calls = tool_calls[:MAX_TOOL_CALLS_PER_TURN]
                                if websocket:
                                    await manager.send_personal_message(
                                        json.dumps({
                                            "type": "progress",
                                            "message": f"Running the first {MAX_TOOL_CALLS_PER_TURN} tool(s) now; will request others if needed.",
                                            "progress": 25,
                                            "step": "tool_execution",
                                            "current_step": 1,
                                            "total_steps": MAX_TOOL_CALLS_PER_TURN
                                        }),
                                        websocket
                                    )
                            
                            tool_sem = asyncio.Semaphore(TOOL_CONCURRENCY)
                            _inflight: Dict[tuple, asyncio.Task] = {}

                            def trim_tool_result(content, max_chars=4000):
                                """Trim verbose tool results to prevent token overflow"""
                                def normalize_text(text):
                                    """Normalize text by stripping HTML, logs, and binary content"""
                                    # Remove HTML tags
                                    text = re.sub(r'<[^>]+>', '', text)
                                    # Remove log patterns (timestamps, log levels)
                                    text = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*?(?:INFO|DEBUG|WARN|ERROR|FATAL).*?\n', '', text)
                                    # Remove binary-like content (base64, hex strings)
                                    text = re.sub(r'[A-Za-z0-9+/]{50,}={0,2}', '[binary_data]', text)
                                    # Remove excessive whitespace
                                    text = re.sub(r'\n\s*\n', '\n\n', text)
                                    return text.strip()
                                
                                def create_summary(text, max_chars=4000):
                                    """Create a heuristic summary for large content"""
                                    if len(text) <= max_chars:
                                        return text
                                    
                                    # Try to find meaningful sections
                                    lines = text.split('\n')
                                    summary_lines = []
                                    total_chars = 0
                                    
                                    # Look for headers, titles, or important content
                                    for line in lines:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        
                                        # Prioritize lines that look like headers or important content
                                        if (line.isupper() or 
                                            line.startswith('#') or 
                                            line.startswith('Title:') or 
                                            line.startswith('Subject:') or
                                            len(line) < 100 and line.endswith(':')):
                                            if total_chars + len(line) + 1 <= max_chars * 0.7:  # Leave room for content
                                                summary_lines.append(line)
                                                total_chars += len(line) + 1
                                        
                                        # Add some regular content
                                        elif total_chars + len(line) + 1 <= max_chars * 0.8:
                                            summary_lines.append(line)
                                            total_chars += len(line) + 1
                                    
                                    if summary_lines:
                                        summary = '\n'.join(summary_lines)
                                        summary += f"\n\n[Content truncated. Original length: {len(text)} characters]"
                                        return summary
                                    else:
                                        # Fallback: just take the first portion
                                        return text[:max_chars] + "...[truncated]"
                                
                                if isinstance(content, list):
                                    # Handle list of content blocks
                                    trimmed_content = []
                                    total_chars = 0
                                    for block in content:
                                        if isinstance(block, dict) and "text" in block:
                                            text = normalize_text(block["text"])
                                            processed_text = create_summary(text, max_chars)
                                            if total_chars + len(processed_text) > max_chars:
                                                # Truncate this block
                                                remaining = max_chars - total_chars
                                                if remaining > 100:  # Only if we have meaningful content left
                                                    trimmed_text = processed_text[:remaining] + "...[truncated]"
                                                    trimmed_content.append({"type": "text", "text": trimmed_text})
                                                break
                                            else:
                                                trimmed_content.append({"type": "text", "text": processed_text})
                                                total_chars += len(processed_text)
                                        else:
                                            trimmed_content.append(block)
                                    return trimmed_content
                                elif isinstance(content, str):
                                    # Handle string content
                                    normalized_text = normalize_text(content)
                                    return create_summary(normalized_text, max_chars)
                                else:
                                    return content

                            async def execute_tool(tool_content):
                                key = (tool_content.name, json.dumps(tool_content.input, sort_keys=True))
                                if key in _inflight:
                                    return await _inflight[key]

                                async def _run():
                                    async with tool_sem:
                                        try:
                                            result = await session.call_tool(tool_content.name, arguments=tool_content.input)
                                            # Trim the result to prevent token overflow
                                            trimmed_content = trim_tool_result(result.content, tool_result_max_chars)
                                            return {
                                                "type": "tool_result",
                                                "tool_use_id": tool_content.id,
                                                "content": trimmed_content
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
                                if websocket:
                                    progress_percentage = int((completed_tools / total_tools) * 100)
                                    await manager.send_personal_message(
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
                            
                            # Check tokens before follow-up call
                            current_tokens = token_manager.count_messages_tokens(messages, system_prompt)
                            follow_up_check = token_manager.check_limits(session_id, current_tokens)
                            
                            if not follow_up_check["allowed"]:
                                full_response += "\n\n[Response truncated due to token limits]"
                                process_query = False
                                continue
                            
                            # --- Minute-budgeted Anthropic call (FOLLOW-UP) ---
                            # Send progress update for response generation
                            if websocket:
                                await manager.send_personal_message(
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

                            for attempt in range(max_retries + 1):
                                # Budget reservation for each attempt
                                await self._budget.reserve()
                                try:
                                    response = await self.anthropic.messages.create(
                                        max_tokens=self._calculate_max_tokens(selected_model, follow_up_check.get("session_remaining", 2048), "follow_up"),
                                        model=selected_model,
                                        system=system_prompt,
                                        tools=available_tools,
                                        messages=messages
                                    )
                                    if hasattr(response, "usage"):
                                        total_in += getattr(response.usage, "input_tokens", 0)
                                        total_out += getattr(response.usage, "output_tokens", 0)
                                    break
                                except Exception as api_error:
                                    if self._is_rate_limited(api_error) and attempt < max_retries:
                                        delay = self._retry_after_secs(api_error, base_delay * (3 ** attempt) + random.uniform(1, 3))
                                        if websocket:
                                            await manager.send_personal_message(
                                                json.dumps({
                                                    "type": "progress",
                                                    "message": f"Rate limited. Retrying in {delay:.1f}s… (Attempt {attempt + 1}/{max_retries + 1})",
                                                    "progress": 60 + (attempt + 1) / (max_retries + 1) * 20
                                                }),
                                                websocket
                                            )
                                        await asyncio.sleep(delay)
                                        continue
                                    elif attempt >= max_retries:
                                        return {"response": "⚠️ **Rate limit reached. Please try again shortly.**", "tokens_used": 0}
                                    else:
                                        return {"response": f"❌ **API Error: {str(api_error)}**", "tokens_used": 0}
                                finally:
                                    self._budget.release()
                            
                            # Update session timestamp only after successful call
                            self._last_by_session[session_id] = time.time()

                            if len(response.content) == 1 and response.content[0].type == "text":
                                full_response += response.content[0].text
                                process_query = False
                    
                    # Token usage is tracked in the finally block
                    
                    if tools_used:
                        tools_info = "\n\n---\n🔧 **Tools Used:** " + ", ".join(set(tools_used))
                        full_response += tools_info
                    
                    return {
                        "response": full_response,
                        "tokens_used": total_in + total_out
                    }
                    
        except Exception as e:
            self._connection_status = "offline"
            return {"response": f"Error: {str(e)}", "tokens_used": 0}
        finally:
            # Always track token usage, even on errors
            total_tokens_used = total_in + total_out
            if total_tokens_used > 0:
                token_manager.add_usage(session_id, total_tokens_used)

mcp_interface = MCPInterface()

class SharePointManager:
    def __init__(self):
        self.graph_config = {
            'client_id': os.getenv('GRAPH_CLIENT_ID'),
            'client_secret': os.getenv('GRAPH_CLIENT_SECRET'),
            'tenant_id': os.getenv('GRAPH_TENANT_ID'),
            'site_url': os.getenv('SHAREPOINT_SITE_URL', 'broadaxis.sharepoint.com:/sites/RFI-project'),
            'folder_path': os.getenv('SHAREPOINT_FOLDER_PATH', 'Documents')
        }
        # Cache for tokens and site/drive info
        self._token_cache = {'token': None, 'expires_at': 0}
        self._site_cache = {'site_id': None, 'drive_id': None, 'expires_at': 0}
        self._session = requests.Session()
        self._session.timeout = 10

    def get_graph_access_token(self):
        """Get access token for Microsoft Graph API with caching"""
        now = time.time()
        
        # Return cached token if still valid (with 5 min buffer)
        if self._token_cache['token'] and now < self._token_cache['expires_at'] - 300:
            return self._token_cache['token']
        
        try:
            token_url = f"https://login.microsoftonline.com/{self.graph_config['tenant_id']}/oauth2/v2.0/token"
            data = {
                'client_id': self.graph_config['client_id'],
                'client_secret': self.graph_config['client_secret'],
                'scope': 'https://graph.microsoft.com/.default',
                'grant_type': 'client_credentials'
            }

            response = self._session.post(token_url, data=data)
            response.raise_for_status()
            token_data = response.json()
            
            # Cache token with expiration
            self._token_cache['token'] = token_data.get('access_token')
            self._token_cache['expires_at'] = now + token_data.get('expires_in', 3600)
            
            return self._token_cache['token']
        except Exception as e:
            error_handler.log_error(e, {'operation': 'get_graph_access_token'})
            return None

    def _get_site_and_drive_info(self):
        """Get and cache site and drive information"""
        now = time.time()
        
        # Return cached info if still valid (cache for 1 hour)
        if (self._site_cache['site_id'] and self._site_cache['drive_id'] and 
            now < self._site_cache['expires_at']):
            return self._site_cache['site_id'], self._site_cache['drive_id']
        
        access_token = self.get_graph_access_token()
        if not access_token:
            return None, None
        
        headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
        
        try:
            # Get site info
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.graph_config['site_url']}"
            site_response = self._session.get(site_url, headers=headers)
            site_response.raise_for_status()
            site_id = site_response.json()['id']
            
            # Get drive info
            drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
            drive_response = self._session.get(drive_url, headers=headers)
            drive_response.raise_for_status()
            drive_id = drive_response.json()['id']
            
            # Cache the info
            self._site_cache.update({
                'site_id': site_id,
                'drive_id': drive_id,
                'expires_at': now + 3600  # 1 hour cache
            })
            
            return site_id, drive_id
        except Exception as e:
            error_handler.log_error(e, {'operation': 'get_site_and_drive_info'})
            return None, None

    def get_sharepoint_files(self):
        """Get files and folders from SharePoint"""
        try:
            site_id, drive_id = self._get_site_and_drive_info()
            if not site_id or not drive_id:
                return {"status": "error", "message": "Failed to get site/drive information", "files": []}

            access_token = self.get_graph_access_token()
            headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}

            # Get files from root folder
            files_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"
            files_response = self._session.get(files_url, headers=headers)

            if files_response.status_code != 200:
                return {"status": "error", "message": "Failed to get SharePoint files", "files": []}

            files_data = files_response.json()
            sharepoint_files = []

            # Process items efficiently
            for item in files_data.get('value', []):
                file_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'type': 'folder' if 'folder' in item else 'file',
                    'size': item.get('size', 0),
                    'modified': item.get('lastModifiedDateTime', ''),
                    'download_url': item.get('@microsoft.graph.downloadUrl', ''),
                    'web_url': item.get('webUrl', ''),
                    'path': item.get('parentReference', {}).get('path', '') + '/' + item['name']
                }

                # Only get folder contents if needed (avoid unnecessary API calls)
                if file_info['type'] == 'folder':
                    folder_files = self.get_folder_contents(site_id, drive_id, item['id'], headers)
                    file_info['children'] = folder_files

                sharepoint_files.append(file_info)

            return {
                "status": "success",
                "message": f"Successfully retrieved {len(sharepoint_files)} items from SharePoint",
                "files": sharepoint_files
            }

        except Exception as e:
            error_handler.log_error(e, {'operation': 'get_sharepoint_files'})
            return {"status": "error", "message": str(e), "files": []}

    def get_folder_contents(self, site_id, drive_id, folder_id, headers):
        """Get contents of a specific folder"""
        try:
            folder_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{folder_id}/children"
            folder_response = requests.get(folder_url, headers=headers)

            if folder_response.status_code != 200:
                return []

            folder_data = folder_response.json()
            folder_files = []

            for item in folder_data.get('value', []):
                file_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'type': 'folder' if 'folder' in item else 'file',
                    'size': item.get('size', 0),
                    'modified': item.get('lastModifiedDateTime', ''),
                    'download_url': item.get('@microsoft.graph.downloadUrl', ''),
                    'web_url': item.get('webUrl', ''),
                    'path': item.get('parentReference', {}).get('path', '') + '/' + item['name']
                }
                folder_files.append(file_info)

            return folder_files

        except Exception as e:
            print(f"Error getting folder contents: {e}")
            return []

    def get_folder_contents_by_path(self, folder_path: str):
        """Get contents of a specific folder by path"""
        try:
            site_id, drive_id = self._get_site_and_drive_info()
            if not site_id or not drive_id:
                return {"status": "error", "message": "Failed to get site/drive information", "files": []}

            access_token = self.get_graph_access_token()
            headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}

            # Get files from the specific folder path
            if folder_path and folder_path != "":
                files_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{folder_path}:/children"
            else:
                files_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"

            files_response = self._session.get(files_url, headers=headers)

            if files_response.status_code != 200:
                return {"status": "error", "message": f"Failed to get folder contents: {files_response.status_code}", "files": []}

            files_data = files_response.json()
            folder_items = []

            for item in files_data.get('value', []):
                file_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'type': 'folder' if 'folder' in item else 'file',
                    'size': item.get('size', 0),
                    'modified': item.get('lastModifiedDateTime', ''),
                    'download_url': item.get('@microsoft.graph.downloadUrl', ''),
                    'web_url': item.get('webUrl', ''),
                    'path': f"{folder_path}/{item['name']}" if folder_path else item['name']
                }
                folder_items.append(file_info)

            return {
                "status": "success",
                "message": f"Successfully retrieved {len(folder_items)} items from folder: {folder_path}",
                "files": folder_items
            }

        except Exception as e:
            error_handler.log_error(e, {'operation': 'get_folder_contents_by_path', 'folder_path': folder_path})
            return {"status": "error", "message": str(e), "files": []}

    def upload_file_to_sharepoint(self, file_content: bytes, filename: str, folder_path: str):
        """Upload a file to SharePoint"""
        try:
            access_token = self.get_graph_access_token()
            if not access_token:
                return {"status": "error", "message": "Failed to get access token"}

            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/octet-stream'
            }

            # Get site and drive info
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.graph_config['site_url']}"
            site_response = requests.get(site_url, headers={'Authorization': f'Bearer {access_token}'})

            if site_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint site"}

            site_data = site_response.json()
            site_id = site_data['id']

            # Get drive
            drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
            drive_response = requests.get(drive_url, headers={'Authorization': f'Bearer {access_token}'})

            if drive_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint drive"}

            drive_data = drive_response.json()
            drive_id = drive_data['id']

            # Create folder path if it doesn't exist
            self.create_sharepoint_folder(site_id, drive_id, folder_path, access_token)

            # Upload file
            upload_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{folder_path}/{filename}:/content"
            print(f"Uploading file to: {upload_url}")

            upload_response = requests.put(upload_url, headers=headers, data=file_content)

            if upload_response.status_code in [200, 201]:
                print(f"✅ Successfully uploaded: {filename}")
                return {"status": "success", "message": f"File uploaded: {filename}"}
            else:
                print(f"❌ Upload failed: {upload_response.status_code} - {upload_response.text}")
                return {"status": "error", "message": f"Upload failed: {upload_response.status_code}"}

        except Exception as e:
            print(f"Error uploading file to SharePoint: {e}")
            return {"status": "error", "message": str(e)}

    def create_sharepoint_folder(self, site_id: str, drive_id: str, folder_path: str, access_token: str):
        """Create folder structure in SharePoint if it doesn't exist"""
        try:
            headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}

            # Split path and create each folder level
            path_parts = folder_path.split('/')
            current_path = ""

            for part in path_parts:
                if part:
                    parent_path = current_path if current_path else ""
                    current_path = f"{current_path}/{part}" if current_path else part

                    # Check if folder exists
                    check_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{current_path}"
                    check_response = requests.get(check_url, headers=headers)

                    if check_response.status_code == 404:
                        # Folder doesn't exist, create it
                        if parent_path:
                            create_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{parent_path}:/children"
                        else:
                            create_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"

                        folder_data = {
                            "name": part,
                            "folder": {},
                            "@microsoft.graph.conflictBehavior": "rename"
                        }

                        create_response = requests.post(create_url, headers=headers, json=folder_data)
                        if create_response.status_code in [200, 201]:
                            print(f"✅ Created folder: {current_path}")
                        else:
                            print(f"⚠️ Folder creation warning: {create_response.status_code}")

        except Exception as e:
            print(f"Error creating SharePoint folder: {e}")

    def get_file_content(self, path: str, binary: bool = False) -> dict:
        """Get file content from SharePoint"""
        try:
            access_token = self.get_graph_access_token()
            if not access_token:
                return {"status": "error", "message": "Failed to get access token"}

            headers = {'Authorization': f'Bearer {access_token}'}
            
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.graph_config['site_url']}"
            site_response = requests.get(site_url, headers=headers)
            if site_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint site"}
            
            site_data = site_response.json()
            site_id = site_data['id']
            
            drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
            drive_response = requests.get(drive_url, headers=headers)
            if drive_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint drive"}
            
            drive_data = drive_response.json()
            drive_id = drive_data['id']
            
            file_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{path}:/content"
            file_response = requests.get(file_url, headers=headers)
            
            if file_response.status_code == 200:
                content = file_response.content if binary else file_response.text
                return {"status": "success", "content": content}
            else:
                return {"status": "error", "message": f"File not found: {file_response.status_code}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_files(self, path: str = "") -> dict:
        """List files in SharePoint folder"""
        return self.get_folder_contents_by_path(path)

    def delete_file(self, path: str) -> dict:
        """Delete a file from SharePoint"""
        try:
            access_token = self.get_graph_access_token()
            if not access_token:
                return {"status": "error", "message": "Failed to get access token"}

            headers = {'Authorization': f'Bearer {access_token}'}
            
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.graph_config['site_url']}"
            site_response = requests.get(site_url, headers=headers)
            if site_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint site"}
            
            site_data = site_response.json()
            site_id = site_data['id']
            
            drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
            drive_response = requests.get(drive_url, headers=headers)
            if drive_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint drive"}
            
            drive_data = drive_response.json()
            drive_id = drive_data['id']
            
            delete_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{path}"
            delete_response = requests.delete(delete_url, headers=headers)
            
            if delete_response.status_code == 204:
                return {"status": "success", "message": "File deleted successfully"}
            else:
                return {"status": "error", "message": f"Delete failed: {delete_response.status_code}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search_files(self, query: str, path: str = "") -> dict:
        """Search for files in SharePoint"""
        try:
            access_token = self.get_graph_access_token()
            if not access_token:
                return {"status": "error", "message": "Failed to get access token"}

            headers = {'Authorization': f'Bearer {access_token}'}
            
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.graph_config['site_url']}"
            site_response = requests.get(site_url, headers=headers)
            if site_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint site"}
            
            site_data = site_response.json()
            site_id = site_data['id']
            
            drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
            drive_response = requests.get(drive_url, headers=headers)
            if drive_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint drive"}
            
            drive_data = drive_response.json()
            drive_id = drive_data['id']
            
            search_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/search(q='{query}')"
            search_response = requests.get(search_url, headers=headers)
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                files = []
                for item in search_data.get('value', []):
                    files.append({
                        'id': item['id'],
                        'name': item['name'],
                        'type': 'folder' if 'folder' in item else 'file',
                        'size': item.get('size', 0),
                        'modified': item.get('lastModifiedDateTime', ''),
                        'path': item.get('parentReference', {}).get('path', '') + '/' + item['name']
                    })
                return {"status": "success", "files": files}
            else:
                return {"status": "error", "message": f"Search failed: {search_response.status_code}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def save_link_to_sharepoint(self, link_url: str, link_title: str, folder_path: str):
        """Save a link as a text file to SharePoint"""
        try:
            # Create text file content
            link_content = f"RFP/RFI/RFQ Link\n"
            link_content += f"Title: {link_title}\n"
            link_content += f"URL: {link_url}\n"
            link_content += f"Extracted: {datetime.now().isoformat()}\n"

            # Clean filename
            safe_filename = self.clean_filename(f"{link_title}.txt")

            # Upload as text file
            return self.upload_file_to_sharepoint(
                link_content.encode('utf-8'),
                safe_filename,
                folder_path
            )

        except Exception as e:
            print(f"Error saving link to SharePoint: {e}")
            return {"status": "error", "message": str(e)}

    def clean_filename(self, filename: str) -> str:
        """Clean filename for SharePoint compatibility"""
        import re
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '-', filename)
        # Replace spaces with hyphens
        filename = re.sub(r'\s+', '-', filename)
        # Remove multiple consecutive hyphens
        filename = re.sub(r'-+', '-', filename)
        # Remove leading/trailing hyphens
        filename = filename.strip('-')
        # Limit length
        if len(filename) > 100:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = f"{name[:95]}.{ext}" if ext else name[:100]
        return filename

class EmailFetcher:
    def __init__(self):
        self.attachments_dir = Path("email_attachments")
        self.attachments_dir.mkdir(exist_ok=True)

        # RFP/RFI/RFQ keywords to search for - using set for O(1) lookup
        self.rfp_keywords = {
            'rfp', 'rfi', 'rfq', 'request for proposal', 'request for information',
            'request for quotation', 'proposal', 'bid', 'tender', 'procurement',
            'solicitation', 'quote', 'quotation'
        }

        # Microsoft Graph API configuration
        self.graph_config = {
            'client_id': os.getenv('GRAPH_CLIENT_ID'),
            'client_secret': os.getenv('GRAPH_CLIENT_SECRET'),
            'tenant_id': os.getenv('GRAPH_TENANT_ID'),
            'user_emails': [
                os.getenv('GRAPH_USER_EMAIL_1'),
                os.getenv('GRAPH_USER_EMAIL_2'),
                os.getenv('GRAPH_USER_EMAIL_3')
            ]
        }

        # Filter out None values efficiently
        self.graph_config['user_emails'] = [email for email in self.graph_config['user_emails'] if email]

        # Email configurations from environment
        self.email_configs = {
            'gmail': {
                'email': os.getenv('GMAIL_EMAIL'),
                'password': os.getenv('GMAIL_PASSWORD'),
                'imap_server': os.getenv('GMAIL_IMAP_SERVER', 'imap.gmail.com'),
                'imap_port': int(os.getenv('GMAIL_IMAP_PORT', 993))
            },
            'outlook': {
                'email': os.getenv('OUTLOOK_EMAIL'),
                'password': os.getenv('OUTLOOK_PASSWORD'),
                'imap_server': os.getenv('OUTLOOK_IMAP_SERVER', 'outlook.office365.com'),
                'imap_port': int(os.getenv('OUTLOOK_IMAP_PORT', 993))
            },
            'corporate': {
                'email': os.getenv('CORPORATE_EMAIL'),
                'password': os.getenv('CORPORATE_PASSWORD'),
                'imap_server': os.getenv('CORPORATE_IMAP_SERVER'),
                'imap_port': int(os.getenv('CORPORATE_IMAP_PORT', 993))
            }
        }
        
        # Cache for Graph API token
        self._token_cache = {'token': None, 'expires_at': 0}
        self._session = requests.Session()
        self._session.timeout = 15

    def has_rfp_keywords(self, text: str) -> bool:
        """Check if text contains RFP/RFI/RFQ related keywords - optimized with set lookup"""
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.rfp_keywords)

    def extract_links_from_email(self, email_content: str) -> List[dict]:
        """Extract relevant links from email content"""
        links = []
        seen_urls = set()  # Track URLs to avoid duplicates

        # Regular expression to find URLs
        url_pattern = r'https?://[^\s<>"{}|\\^\[\]]+[^\s<>"{}|\\^\[\].,;:!?]'
        found_urls = re.findall(url_pattern, email_content, re.IGNORECASE)

        # Keywords that suggest RFP/procurement related links
        rfp_link_keywords = [
            'rfp', 'rfi', 'rfq', 'proposal', 'bid', 'tender', 'procurement',
            'solicitation', 'quote', 'quotation', 'contract', 'vendor',
            'supplier', 'opportunity', 'award', 'government', 'portal'
        ]

        for url in found_urls:
            try:
                # Clean up the URL
                url = url.strip()

                # Skip if we've already seen this URL
                if url in seen_urls:
                    continue

                # Parse URL to get domain and path
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                path = parsed.path.lower()

                # Check if URL is likely RFP/procurement related
                is_rfp_related = False

                # Check domain and path for RFP keywords
                full_url_text = f"{domain} {path}".lower()
                if any(keyword in full_url_text for keyword in rfp_link_keywords):
                    is_rfp_related = True

                # Check for government domains (often have procurement portals)
                gov_domains = ['.gov', '.mil', '.edu', 'procurement', 'tender', 'bid']
                if any(gov_domain in domain for gov_domain in gov_domains):
                    is_rfp_related = True

                # Check for common procurement platforms
                procurement_platforms = [
                    'sam.gov', 'fedbizopps', 'grants.gov', 'beta.sam.gov',
                    'merx.com', 'biddingo.com', 'demandstar.com', 'publicsector.ca',
                    'bonfirehub.com', 'ionwave.net', 'questcdn.com'
                ]
                if any(platform in domain for platform in procurement_platforms):
                    is_rfp_related = True

                if is_rfp_related:
                    # Add to seen URLs to avoid duplicates
                    seen_urls.add(url)

                    # Try to get a meaningful title from the URL
                    link_title = self.generate_link_title(url, domain, path)

                    links.append({
                        'url': url,
                        'title': link_title,
                        'domain': domain,
                        'type': 'rfp_link'
                    })

            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                continue

        return links
    
    def generate_link_title(self, url: str, domain: str, path: str) -> str:
        """Generate a meaningful title for the link"""
        # Try to extract meaningful parts from the URL
        if 'sam.gov' in domain:
            return f"SAM.gov Opportunity - {domain}"
        elif '.gov' in domain:
            return f"Government Procurement - {domain}"
        elif 'rfp' in path or 'rfp' in domain:
            return f"RFP Portal - {domain}"
        elif 'rfi' in path or 'rfi' in domain:
            return f"RFI Portal - {domain}"
        elif 'rfq' in path or 'rfq' in domain:
            return f"RFQ Portal - {domain}"
        elif 'tender' in path or 'tender' in domain:
            return f"Tender Portal - {domain}"
        elif 'bid' in path or 'bid' in domain:
            return f"Bidding Portal - {domain}"
        elif 'procurement' in path or 'procurement' in domain:
            return f"Procurement Portal - {domain}"
        else:
            return f"Opportunity Link - {domain}"

    def clean_filename(self, filename: str) -> str:
        """Clean filename for SharePoint compatibility"""
        import re
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '-', filename)
        # Replace spaces with hyphens
        filename = re.sub(r'\s+', '-', filename)
        # Remove multiple consecutive hyphens
        filename = re.sub(r'-+', '-', filename)
        # Remove leading/trailing hyphens
        filename = filename.strip('-')
        # Limit length
        if len(filename) > 100:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = f"{name[:95]}.{ext}" if ext else name[:100]
        return filename

    def save_attachment(self, attachment_data: bytes, filename: str, email_date: str) -> dict:
        """Save email attachment to local folder"""
        try:
            # Create date-based subfolder
            date_folder = self.attachments_dir / email_date.split()[0]  # YYYY-MM-DD
            date_folder.mkdir(exist_ok=True)

            # Clean filename
            clean_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            file_path = date_folder / clean_filename

            # Save file
            with open(file_path, 'wb') as f:
                f.write(attachment_data)

            return {
                "filename": clean_filename,
                "file_path": str(file_path),
                "file_size": len(attachment_data),
                "download_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error saving attachment {filename}: {e}")
            return None

    def get_graph_access_token(self) -> str:
        """Get access token for Microsoft Graph API with caching"""
        now = time.time()
        
        # Return cached token if still valid (with 5 min buffer)
        if self._token_cache['token'] and now < self._token_cache['expires_at'] - 300:
            return self._token_cache['token']
        
        try:
            token_url = f"https://login.microsoftonline.com/{self.graph_config['tenant_id']}/oauth2/v2.0/token"

            token_data = {
                'grant_type': 'client_credentials',
                'client_id': self.graph_config['client_id'],
                'client_secret': self.graph_config['client_secret'],
                'scope': 'https://graph.microsoft.com/.default'
            }

            response = self._session.post(token_url, data=token_data)
            response.raise_for_status()
            
            token_response = response.json()
            access_token = token_response['access_token']
            
            # Cache token with expiration
            self._token_cache['token'] = access_token
            self._token_cache['expires_at'] = now + token_response.get('expires_in', 3600)
            
            return access_token
        except requests.exceptions.Timeout:
            error_handler.log_error(Exception("Timeout getting Graph access token"), {'operation': 'get_graph_access_token'})
            return None
        except requests.exceptions.RequestException as e:
            error_handler.log_error(e, {'operation': 'get_graph_access_token'})
            return None
        except Exception as e:
            error_handler.log_error(e, {'operation': 'get_graph_access_token'})
            return None

    def fetch_emails_graph(self) -> dict:
        """Fetch emails using Microsoft Graph API from multiple accounts"""
        if not self.graph_config['client_id'] or not self.graph_config['client_secret'] or not self.graph_config['tenant_id']:
            return {
                "status": "error",
                "message": "Microsoft Graph API configuration incomplete. Please check GRAPH_CLIENT_ID, GRAPH_CLIENT_SECRET, and GRAPH_TENANT_ID in .env file.",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }

        if not self.graph_config['user_emails']:
            return {
                "status": "error",
                "message": "No email accounts configured. Please check GRAPH_USER_EMAIL_1, GRAPH_USER_EMAIL_2, GRAPH_USER_EMAIL_3 in .env file.",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }

        try:
            # Get access token
            access_token = self.get_graph_access_token()
            if not access_token:
                return {
                    "status": "error",
                    "message": "Failed to get Microsoft Graph access token. Check your client credentials.",
                    "emails_found": 0,
                    "attachments_downloaded": 0,
                    "fetched_emails": []
                }

            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }

            fetched_emails = []
            total_attachments = 0

            # Process all email accounts efficiently
            for user_email in self.graph_config['user_emails']:
                # Get emails from the last 30 days for this account
                graph_url = f"https://graph.microsoft.com/v1.0/users/{user_email}/messages"
                params = {
                    '$top': 50,  # Get more emails to filter through
                    '$select': 'id,subject,sender,receivedDateTime,hasAttachments,body',
                    '$orderby': 'receivedDateTime desc'
                }

                try:
                    response = self._session.get(graph_url, headers=headers, params=params)
                    response.raise_for_status()
                    emails_data = response.json()
                except requests.exceptions.Timeout:
                    error_handler.log_error(Exception(f"Timeout fetching emails from {user_email}"), 
                                          {'operation': 'fetch_emails_graph', 'user_email': user_email})
                    continue  # Skip this account and try the next one
                except requests.exceptions.RequestException as e:
                    error_handler.log_error(e, {'operation': 'fetch_emails_graph', 'user_email': user_email})
                    continue  # Skip this account and try the next one

                # Process each email and filter for RFP/RFI/RFQ keywords
                for email_item in emails_data.get('value', []):
                    # Check if email has RFP keywords first
                    has_keywords = self.has_rfp_keywords(email_item.get('subject', ''))

                    if has_keywords:
                        attachments = []

                        # Extract email content for link detection
                        email_content = ""
                        if email_item.get('body') and email_item['body'].get('content'):
                            email_content = email_item['body']['content']

                        # Extract links from email content
                        extracted_links = self.extract_links_from_email(email_content)

                        # Create SharePoint folder path for this email
                        email_date = email_item['receivedDateTime'][:10]  # YYYY-MM-DD
                        email_subject_clean = self.clean_filename(email_item['subject'])
                        email_folder_name = f"{email_date}_{email_subject_clean}"
                        sharepoint_folder_path = f"Emails/{user_email}/{email_folder_name}"

                        # Initialize SharePoint manager for uploads
                        sharepoint_manager = SharePointManager()

                        # Add links as "link attachments" and save to SharePoint
                        for link in extracted_links:
                            # Save link to SharePoint
                            link_result = sharepoint_manager.save_link_to_sharepoint(
                                link['url'],
                                link['title'],
                                sharepoint_folder_path
                            )

                            # Link saved to SharePoint

                            link_attachment = {
                                'filename': link['title'],
                                'file_path': '',  # Empty string for links
                                'file_size': 0,   # Zero for links
                                'url': link['url'],
                                'domain': link['domain'],
                                'type': 'link',
                                'download_date': datetime.now().isoformat(),
                                'sharepoint_path': sharepoint_folder_path
                            }
                            attachments.append(link_attachment)
                            total_attachments += 1

                        # Process file attachments efficiently
                        if email_item.get('hasAttachments'):
                            # Get attachments for this specific user account
                            attachments_url = f"https://graph.microsoft.com/v1.0/users/{user_email}/messages/{email_item['id']}/attachments"
                            attachments_response = self._session.get(attachments_url, headers=headers)

                            if attachments_response.status_code == 200:
                                attachments_data = attachments_response.json()
                                date_str = email_item['receivedDateTime'][:10]  # Get YYYY-MM-DD part

                                for attachment in attachments_data.get('value', []):
                                    if attachment.get('@odata.type') == '#microsoft.graph.fileAttachment':
                                        # Download attachment
                                        attachment_content = base64.b64decode(attachment['contentBytes'])

                                        # Save to local storage and SharePoint in parallel
                                        saved_attachment = self.save_attachment(
                                            attachment_content,
                                            attachment['name'],
                                            date_str
                                        )

                                        if saved_attachment:
                                            # Upload to SharePoint
                                            clean_filename = self.clean_filename(attachment['name'])
                                            sharepoint_manager.upload_file_to_sharepoint(
                                                attachment_content,
                                                clean_filename,
                                                sharepoint_folder_path
                                            )

                                            saved_attachment['type'] = 'file'  # Mark as file attachment
                                            saved_attachment['sharepoint_path'] = sharepoint_folder_path
                                            attachments.append(saved_attachment)
                                            total_attachments += 1

                        # Make a deep copy of attachments to avoid reference sharing
                        import copy
                        email_attachments = copy.deepcopy(attachments)

                        fetched_emails.append({
                            "email_id": email_item['id'],
                            "sender": email_item['sender']['emailAddress']['address'],
                            "subject": email_item['subject'],
                            "date": email_item['receivedDateTime'],
                            "account": user_email,  # Use the current user_email being processed
                            "attachments": email_attachments,
                            "has_rfp_keywords": has_keywords
                        })

            account_count = len(self.graph_config['user_emails'])
            return {
                "status": "success",
                "message": f"Successfully fetched {len(fetched_emails)} RFP/RFI/RFQ emails from {account_count} email accounts using Microsoft Graph API",
                "emails_found": len(fetched_emails),
                "attachments_downloaded": total_attachments,
                "fetched_emails": fetched_emails
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Microsoft Graph API error: {str(e)}",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching emails via Graph API: {str(e)}",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }
        
    def fetch_emails_real(self, email_account: str = 'gmail') -> dict:
        """Fetch emails from real email account"""
        config = self.email_configs.get(email_account)

        if not config or not config['email'] or not config['password']:
            return {
                "status": "error",
                "message": f"Email configuration for {email_account} not found or incomplete. Please check your .env file.",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }

        try:
            # Connect to email server
            mail = imaplib.IMAP4_SSL(config['imap_server'], config['imap_port'])

            # Try different authentication methods for corporate accounts
            try:
                mail.login(config['email'], config['password'])
            except imaplib.IMAP4.error:
                # Try with username only (without domain)
                username = config['email'].split('@')[0]
                mail.login(username, config['password'])
            mail.select('inbox')

            # Search for emails with RFP/RFI/RFQ keywords
            search_criteria = []
            for keyword in self.rfp_keywords:
                search_criteria.append(f'SUBJECT "{keyword}"')

            # Search in the last 30 days
            search_query = f'(SINCE "01-Jan-2025") ({" OR ".join(search_criteria)})'

            result, message_numbers = mail.search(None, search_query)

            if result != 'OK':
                return {
                    "status": "error",
                    "message": "Failed to search emails",
                    "emails_found": 0,
                    "attachments_downloaded": 0,
                    "fetched_emails": []
                }

            email_ids = message_numbers[0].split()
            fetched_emails = []
            total_attachments = 0

            # Process each email
            for email_id in email_ids[-10:]:  # Limit to last 10 emails
                result, message_data = mail.fetch(email_id, '(RFC822)')

                if result == 'OK':
                    email_message = email.message_from_bytes(message_data[0][1])

                    # Extract email details
                    sender = email_message.get('From', 'Unknown')
                    subject = email_message.get('Subject', 'No Subject')
                    date = email_message.get('Date', 'Unknown')

                    # Check if email has RFP keywords
                    has_keywords = self.has_rfp_keywords(subject)

                    if has_keywords:
                        attachments = []

                        # Process attachments
                        for part in email_message.walk():
                            if part.get_content_disposition() == 'attachment':
                                filename = part.get_filename()
                                if filename:
                                    attachment_data = part.get_payload(decode=True)
                                    saved_attachment = self.save_attachment(attachment_data, filename, date)
                                    if saved_attachment:
                                        attachments.append(saved_attachment)
                                        total_attachments += 1

                        # Make a deep copy of attachments to avoid reference sharing
                        import copy
                        email_attachments = copy.deepcopy(attachments)

                        fetched_emails.append({
                            "email_id": email_id.decode(),
                            "sender": sender,
                            "subject": subject,
                            "date": date,
                            "account": config['email'],
                            "attachments": email_attachments,
                            "has_rfp_keywords": has_keywords
                        })

            mail.close()
            mail.logout()

            return {
                "status": "success",
                "message": f"Successfully fetched {len(fetched_emails)} RFP/RFI/RFQ emails from {config['email']}",
                "emails_found": len(fetched_emails),
                "attachments_downloaded": total_attachments,
                "fetched_emails": fetched_emails
            }

        except imaplib.IMAP4.error as e:
            return {
                "status": "error",
                "message": f"IMAP error: {str(e)}. Check your email credentials and server settings.",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching emails: {str(e)}",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }

    async def fetch_emails_demo(self) -> EmailFetchResponse:
        """Demo implementation - simulates email fetching"""
        # Simulate finding RFP-related emails
        demo_emails = [
            {
                "email_id": "email_001",
                "sender": "procurement@techcorp.com",
                "subject": "RFP for Cloud Infrastructure Services - Due March 15",
                "date": "2025-01-15 10:30:00",
                "account": "proposals@broadaxis.com",
                "attachments": [
                    {
                        "filename": "RFP_Cloud_Infrastructure_2025.pdf",
                        "file_path": "email_attachments/2025-01-15/RFP_Cloud_Infrastructure_2025.pdf",
                        "file_size": 2048576,
                        "download_date": datetime.now().isoformat()
                    }
                ],
                "has_rfp_keywords": True
            },
            {
                "email_id": "email_002",
                "sender": "sourcing@govagency.gov",
                "subject": "RFI - Software Development Services",
                "date": "2025-01-14 14:20:00",
                "account": "rfp.team@broadaxis.com",
                "attachments": [
                    {
                        "filename": "RFI_Software_Development.docx",
                        "file_path": "email_attachments/2025-01-14/RFI_Software_Development.docx",
                        "file_size": 1024000,
                        "download_date": datetime.now().isoformat()
                    }
                ],
                "has_rfp_keywords": True
            },
            {
                "email_id": "email_003",
                "sender": "purchasing@enterprise.com",
                "subject": "RFQ for Hardware Procurement - Urgent",
                "date": "2025-01-13 09:15:00",
                "account": "business@broadaxis.com",
                "attachments": [
                    {
                        "filename": "Hardware_RFQ_Specifications.pdf",
                        "file_path": "email_attachments/2025-01-13/Hardware_RFQ_Specifications.pdf",
                        "file_size": 3072000,
                        "download_date": datetime.now().isoformat()
                    }
                ],
                "has_rfp_keywords": True
            }
        ]

        # Create demo attachment files
        for email_data in demo_emails:
            for attachment in email_data["attachments"]:
                file_path = Path(attachment["file_path"])
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Create demo file content
                demo_content = f"""Demo {attachment['filename']}

This is a simulated RFP/RFI/RFQ document downloaded from email.
Email: {email_data['sender']}
Subject: {email_data['subject']}
Date: {email_data['date']}

[This would contain the actual RFP/RFI/RFQ content in a real scenario]
"""
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(demo_content)

        return {
            "status": "success",
            "message": f"Successfully fetched {len(demo_emails)} RFP/RFI/RFQ emails",
            "emails_found": len(demo_emails),
            "attachments_downloaded": sum(len(email_data["attachments"]) for email_data in demo_emails),
            "fetched_emails": demo_emails
        }

# Initialize email fetcher
email_fetcher = EmailFetcher()

# Global storage for real fetched emails (in production, use database)
real_fetched_emails = []

def save_real_emails_to_file(emails):
    """Save real fetched emails to a JSON file for persistence"""
    try:
        with open('real_fetched_emails.json', 'w') as f:
            json.dump(emails, f, indent=2)
    except Exception as e:
        print(f"Error saving real emails: {e}")

def load_real_emails_from_file():
    """Load real fetched emails from JSON file"""
    try:
        if os.path.exists('real_fetched_emails.json'):
            with open('real_fetched_emails.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading real emails: {e}")
    return []

# Load real emails on startup
real_fetched_emails = load_real_emails_from_file()

# Session file storage
session_files = {}

async def run_mcp_query(query: str, enabled_tools: List[str] = None, model: str = None, session_id: str = "default", websocket = None) -> Dict:
    return await mcp_interface.process_query_with_anthropic(query, enabled_tools, model, session_id, websocket)

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
        "initializing": mcp_interface._initializing,
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

@app.get("/api/model-info/{model}")
def get_model_info(model: str):
    """Get information about a specific model's configuration."""
    try:
        model_info = mcp_interface.get_model_info(model)
        return {
            "model_info": model_info,
            "status": "success"
        }
    except Exception as e:
        error_handler.log_error(e, {'model': model, 'operation': 'get_model_info'})
        return {
            "error": f"Failed to get model info for {model}",
            "status": "error"
        }

@app.get("/api/models")
def get_available_models():
    """Get list of all available models and their configurations."""
    try:
        models = {}
        for model_name in mcp_interface._model_configs.keys():
            models[model_name] = mcp_interface.get_model_info(model_name)
        
        return {
            "models": models,
            "status": "success"
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_available_models'})
        return {
            "error": "Failed to get available models",
            "status": "error"
        }

@app.get("/api/test-token-calculation")
def test_token_calculation():
    """Test endpoint to verify adaptive token calculation logic."""
    try:
        test_cases = [
            {"model": "claude-3-7-sonnet-20250219", "session_remaining": 1000, "call_type": "first"},
            {"model": "claude-3-7-sonnet-20250219", "session_remaining": 1000, "call_type": "follow_up"},
            {"model": "claude-3-7-sonnet-20250219", "session_remaining": 10000, "call_type": "first"},
            {"model": "claude-3-7-sonnet-20250219", "session_remaining": 10000, "call_type": "follow_up"},
            {"model": "claude-sonnet-4-20250514", "session_remaining": 5000, "call_type": "first"},
            {"model": "claude-opus-4-20250514", "session_remaining": 3000, "call_type": "follow_up"},
        ]
        
        results = {}
        for case in test_cases:
            max_tokens = mcp_interface._calculate_max_tokens(
                case["model"], 
                case["session_remaining"], 
                case["call_type"]
            )
            results[f"{case['model']}_{case['call_type']}_{case['session_remaining']}"] = {
                "input": case,
                "output": max_tokens
            }
        
        return {
            "test_results": results,
            "status": "success"
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'test_token_calculation'})
        return {
            "error": "Failed to test token calculation",
            "status": "error"
        }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = "default"):
    """Upload and process PDF files using MCP tool"""
    if not file.filename:
        raise ValidationError("No file provided")
    
    # Handle null session_id from frontend
    if session_id == "null" or not session_id:
        session_id = "default"
    
    # Validate file type
    allowed_extensions = {'.pdf', '.txt', '.md', '.docx', '.doc'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_extensions:
        raise ValidationError(f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}")
    
    # Check session file limit
    session_key = f"session_{session_id}"
    if session_key in session_files and len(session_files[session_key]) >= 3:
        raise ValidationError("Maximum 3 files per session")
    
    try:
        file_content = await file.read()
        
        if not file_content:
            raise ValidationError("Uploaded file is empty")
        
        if len(file_content) > 50 * 1024 * 1024:  # 50MB limit
            raise ValidationError("File size exceeds 50MB limit")
        
        # Use MCP tool to process the document
        async with stdio_client(mcp_interface.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Call document processing tool
                result = await session.call_tool(
                    "process_uploaded_document",
                    arguments={
                        "file_content": base64.b64encode(file_content).decode('utf-8'),
                        "filename": file.filename,
                        "session_id": session_id
                    }
                )
                
                # Store file info in session
                if session_key not in session_files:
                    session_files[session_key] = []
                
                session_files[session_key].append({
                    "filename": file.filename,
                    "size": len(file_content),
                    "upload_time": datetime.now().isoformat()
                })
                
                return {
                    "status": "success",
                    "filename": file.filename,
                    "size": len(file_content),
                    "files_in_session": len(session_files[session_key]),
                    "message": result.content[0].text if result.content else "Document processed successfully"
                }
        
    except BroadAxisError:
        raise
    except Exception as e:
        error_handler.log_error(e, {'filename': file.filename, 'operation': 'file_upload'})
        raise FileOperationError("Failed to process uploaded file", {'filename': file.filename})

@app.post("/api/chat/document")
async def chat_with_document(request: ChatRequest, session_id: str = "default", filename: str = None):
    """Chat with uploaded documents using hybrid retrieval"""
    if not request.query or not request.query.strip():
        raise ValidationError("Query cannot be empty")
    
    # Handle null session_id from frontend
    if session_id == "null" or not session_id:
        session_id = "default"
    
    try:
        # Check if session has uploaded files
        session_key = f"session_{session_id}"
        if session_key not in session_files or len(session_files[session_key]) == 0:
            raise ValidationError("No documents uploaded for this session")
        
        # Process query with document context
        result = await run_mcp_query(
            query=request.query.strip(),
            enabled_tools=request.enabled_tools,
            model=request.model,
            session_id=session_id
        )
        
        if not isinstance(result, dict) or "response" not in result:
            raise ExternalAPIError("Invalid response from MCP query processor")
        
        usage = token_manager.get_usage(session_id)
        return {
            "response": result["response"],
            "status": "success",
            "tokens_used": result.get("tokens_used", 0),
            "tokens_remaining": usage["session_limit"] - usage["session_used"],
            "usage": usage
        }
        
    except BroadAxisError:
        raise
    except Exception as e:
        error_handler.log_error(e, {'session_id': session_id, 'query_length': len(request.query)})
        raise ExternalAPIError("Failed to process document chat request", {'original_error': str(e)})

@app.post("/api/initialize")
async def initialize_mcp():
    """Initialize MCP server - fetch both tools and prompts"""
    try:
        await mcp_interface.initialize()
        return {
            "tools": mcp_interface._tools_cache or [],
            "prompts": mcp_interface._prompts_cache or [],
            "status": "success",
            "connection_status": mcp_interface._connection_status
        }
    except BroadAxisError:
        raise
    except Exception as e:
        error_handler.log_error(e, {'operation': 'initialize_mcp'})
        return {
            "tools": [],
            "prompts": [],
            "status": "error",
            "connection_status": "offline",
            "error_message": "Failed to initialize MCP server"
        }




@app.get("/api/test-graph-auth")
async def test_graph_auth():
    """Test Microsoft Graph API authentication"""
    try:
        print("🧪 Testing Microsoft Graph API authentication...")
        
        # Test access token
        token = email_fetcher.get_graph_access_token()
        if not token:
            return {
                "status": "error",
                "message": "Failed to get access token",
                "step": "authentication"
            }
        
        print("✅ Access token obtained")
        
        # Test basic Graph API call using application permissions
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        
        # Test with first configured email account
        user_emails = [
            os.getenv('GRAPH_USER_EMAIL_1'),
            os.getenv('GRAPH_USER_EMAIL_2'),
            os.getenv('GRAPH_USER_EMAIL_3')
        ]
        user_emails = [email for email in user_emails if email]
        
        if not user_emails:
            return {
                "status": "error",
                "message": "No email accounts configured. Check GRAPH_USER_EMAIL_1, GRAPH_USER_EMAIL_2, GRAPH_USER_EMAIL_3 in .env file.",
                "step": "config_check"
            }
        
        test_email = user_emails[0]
        test_url = f"https://graph.microsoft.com/v1.0/users/{test_email}"
        
        response = requests.get(test_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_info = response.json()
            return {
                "status": "success",
                "message": f"Microsoft Graph API authentication working! Connected to {user_info.get('displayName', test_email)}",
                "user_info": {
                    "email": user_info.get('mail', test_email),
                    "displayName": user_info.get('displayName', 'Unknown'),
                    "configured_accounts": len(user_emails)
                }
            }
        elif response.status_code == 403:
            return {
                "status": "error",
                "message": "Microsoft Graph API permissions insufficient. Your app needs 'Mail.Read' and 'User.Read.All' application permissions. Contact your Azure admin to grant these permissions.",
                "step": "permissions",
                "test_email": test_email,
                "fix_instructions": [
                    "1. Go to Azure Portal > App Registrations",
                    "2. Find your app and go to API Permissions",
                    "3. Add Microsoft Graph Application permissions:",
                    "   - Mail.Read (to read emails)",
                    "   - User.Read.All (to access user info)",
                    "4. Click 'Grant admin consent'",
                    "5. Wait 5-10 minutes for permissions to propagate"
                ]
            }
        else:
            return {
                "status": "error",
                "message": f"Graph API test failed: {response.status_code} - {response.text[:200]}",
                "step": "api_test",
                "test_email": test_email
            }
            
    except Exception as e:
        print(f"❌ Graph API test failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "step": "exception"
        }

@app.post("/api/fetch-emails", response_model=EmailFetchResponse)
async def fetch_emails(request: EmailFetchRequest = EmailFetchRequest()):
    """Fetch RFP/RFI/RFQ emails and download attachments"""
    try:
        print("📧 Starting email fetch process...")
        
        # Quick auth test first
        token = email_fetcher.get_graph_access_token()
        if not token:
            print("⚠️ Authentication failed, using demo mode")
            return await email_fetcher.fetch_emails_demo()
        
        print("✅ Authentication successful, testing permissions...")
        
        # Test permissions with a simple API call
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        user_emails = [os.getenv('GRAPH_USER_EMAIL_1'), os.getenv('GRAPH_USER_EMAIL_2'), os.getenv('GRAPH_USER_EMAIL_3')]
        user_emails = [email for email in user_emails if email]
        
        if user_emails:
            test_url = f"https://graph.microsoft.com/v1.0/users/{user_emails[0]}"
            test_response = requests.get(test_url, headers=headers, timeout=5)
            
            if test_response.status_code == 403:
                print("⚠️ Insufficient permissions, using demo mode")
                demo_result = await email_fetcher.fetch_emails_demo()
                demo_result['message'] += " (Demo mode - Microsoft Graph API permissions needed)"
                return demo_result
        
        # Try real email fetching
        result = email_fetcher.fetch_emails_graph()
        
        # If real fetching fails, fall back to demo
        if result['status'] == 'error' and ('403' in result['message'] or 'Authorization' in result['message']):
            print("⚠️ Real email fetch failed due to permissions, using demo mode")
            demo_result = await email_fetcher.fetch_emails_demo()
            demo_result['message'] += " (Demo mode - Microsoft Graph API permissions needed)"
            return demo_result

        # Store real fetched emails for display
        if result['status'] == 'success':
            global real_fetched_emails
            real_fetched_emails = result['fetched_emails']
            # Save to file for persistence
            save_real_emails_to_file(real_fetched_emails)

        return EmailFetchResponse(**result)
    except Exception as e:
        print(f"Error fetching emails: {e}")
        print("⚠️ Falling back to demo mode")
        demo_result = await email_fetcher.fetch_emails_demo()
        demo_result['message'] += " (Demo mode - Error occurred)"
        return demo_result

@app.get("/api/fetched-emails")
async def get_fetched_emails():
    """Get list of previously fetched emails"""
    try:
        # Check if we have real fetched emails from Graph API or IMAP
        global real_fetched_emails
        if real_fetched_emails:
            # Return real fetched emails
            email_accounts = {}
            for email in real_fetched_emails:
                account = email.get('account', 'unknown@example.com')
                if account not in email_accounts:
                    email_accounts[account] = {
                        'count': 0,
                        'emails': [],
                        'latest_subject': '',
                        'latest_date': '',
                        'total_files': 0
                    }

                email_accounts[account]['count'] += 1
                email_accounts[account]['emails'].append(email)
                email_accounts[account]['latest_subject'] = email.get('subject', 'No Subject')
                email_accounts[account]['latest_date'] = email.get('date', '')
                email_accounts[account]['total_files'] += len(email.get('attachments', []))

            # Convert to the expected format
            emails = []
            for i, (account, data) in enumerate(email_accounts.items(), 1):
                emails.append({
                    'id': i,
                    'email': account,
                    'count': data['count'],
                    'latest_subject': data['latest_subject'],
                    'latest_date': data['latest_date'],
                    'latest_files': data['total_files'],
                    'emails': data['emails']
                })

            return {
                'total_count': len(real_fetched_emails),
                'total_files': sum(len(email.get('attachments', [])) for email in real_fetched_emails),
                'emails': emails
            }

        # Fallback to demo data if no real emails fetched
        attachments_dir = Path("email_attachments")
        if not attachments_dir.exists():
            return {"emails": [], "total_count": 0}

        # Count files in attachments directory
        total_files = sum(1 for file_path in attachments_dir.rglob("*") if file_path.is_file())

        demo_emails = [
            {
                "id": 1,
                "email": "proposals@broadaxis.com",
                "count": 8,
                "latest_subject": "RFP for Cloud Infrastructure Services",
                "latest_date": "2025-01-15 10:30:00",
                "latest_file": "RFP_Cloud_Infrastructure_2025.pdf"
            },
            {
                "id": 2,
                "email": "rfp.team@broadaxis.com",
                "count": 5,
                "latest_subject": "RFI - Software Development Services",
                "latest_date": "2025-01-14 14:20:00",
                "latest_file": "RFI_Software_Development.docx"
            },
            {
                "id": 3,
                "email": "business@broadaxis.com",
                "count": 3,
                "latest_subject": "RFQ for Hardware Procurement",
                "latest_date": "2025-01-13 09:15:00",
                "latest_file": "Hardware_RFQ_Specifications.pdf"
            }
        ]

        return {
            "emails": demo_emails,
            "total_count": sum(email["count"] for email in demo_emails),
            "total_files": total_files
        }
    except Exception as e:
        print(f"Error getting fetched emails: {e}")
        return {"emails": [], "total_count": 0, "total_files": 0}

@app.get("/api/email-attachments/{email_id}")
async def get_email_attachments(email_id: int):
    """Get attachments for a specific email account"""
    try:
        # Check if we have real fetched emails
        global real_fetched_emails
        if real_fetched_emails:
            # Group emails by account first
            email_accounts = {}
            for email in real_fetched_emails:
                account = email.get('account', 'unknown@example.com')
                if account not in email_accounts:
                    email_accounts[account] = []
                email_accounts[account].append(email)

            # Convert to list with IDs (same logic as fetched-emails endpoint)
            account_list = []
            for i, (account, emails) in enumerate(email_accounts.items(), 1):
                account_list.append({
                    'id': i,
                    'account': account,
                    'emails': emails
                })

            # Find the specific account for this email_id
            target_account = None
            for account_data in account_list:
                if account_data['id'] == email_id:
                    target_account = account_data
                    break

            if target_account:
                # Return attachments only for this specific account
                account_attachments = []
                for email in target_account['emails']:
                    attachments = email.get('attachments', [])
                    email_subject = email.get('subject', 'No Subject')
                    email_sender = email.get('sender', 'Unknown Sender')
                    email_date = email.get('date', '')

                    # Add email context to each attachment
                    for attachment in attachments:
                        attachment_with_context = attachment.copy()
                        attachment_with_context['email_subject'] = email_subject
                        attachment_with_context['email_sender'] = email_sender
                        attachment_with_context['email_date'] = email_date
                        account_attachments.append(attachment_with_context)

                return {
                    "email_id": email_id,
                    "account": target_account['account'],
                    "attachments": account_attachments
                }

        # Fallback to demo data if no real emails
        attachments_data = {
            1: [
                {"filename": "RFP_Cloud_Infrastructure_2025.pdf", "date": "2025-01-15", "size": "2.0 MB"},
                {"filename": "Technical_Requirements.docx", "date": "2025-01-14", "size": "1.5 MB"},
                {"filename": "Budget_Guidelines.xlsx", "date": "2025-01-13", "size": "0.8 MB"}
            ],
            2: [
                {"filename": "RFI_Software_Development.docx", "date": "2025-01-14", "size": "1.0 MB"},
                {"filename": "Vendor_Questionnaire.pdf", "date": "2025-01-12", "size": "0.5 MB"}
            ],
            3: [
                {"filename": "Hardware_RFQ_Specifications.pdf", "date": "2025-01-13", "size": "3.0 MB"}
            ]
        }

        return {
            "email_id": email_id,
            "attachments": attachments_data.get(email_id, [])
        }
    except Exception as e:
        print(f"Error getting email attachments: {e}")
        return {"email_id": email_id, "attachments": []}


# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication."""
    session_id = f"ws_{id(websocket)}_{int(time.time())}"
    # NEW: per-socket gate to avoid overlapping requests from one client
    if not hasattr(websocket, "__gate"):
        websocket.__gate = asyncio.Semaphore(1)
    
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

                # Handle different message types
                message_type = message_data.get("type", "")
                
                # Handle heartbeat/ping messages
                if message_type in ["ping", "pong", "heartbeat"]:
                    # Send pong response for ping, or just acknowledge
                    if message_type == "ping":
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "pong",
                                "timestamp": datetime.now().isoformat()
                            }),
                            websocket
                        )
                    continue
                
                # Handle connection messages
                if message_type == "connection":
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "connection",
                            "message": "Connection established",
                            "status": "success",
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                    continue
                
                # Handle chat messages (default behavior)
                query = message_data.get("query", "").strip()
                enabled_tools = message_data.get("enabled_tools", [])
                model = message_data.get("model", "claude-3-7-sonnet-20250219")
                
                if not query:
                    # Log the message that caused the error for debugging
                    print(f"Received message without query: {message_data}")
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
                    json.dumps({
                        "type": "status", 
                        "message": "Processing your request...",
                        "status": "info",
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )

                try:
                    # Process query with enabled tools (guarded per-socket)
                    async with websocket.__gate:
                        result = await run_mcp_query(
                            query, enabled_tools, model, session_id, websocket
                        )
                    
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

                    # Send success status
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "status",
                            "message": "Response generated successfully!",
                            "status": "success",
                            "timestamp": datetime.now().isoformat()
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


@app.get("/api/test-sharepoint")
async def test_sharepoint():
    """Test SharePoint connection step by step"""
    try:
        print("🧪 Testing SharePoint connection...")

        # Test 1: Access token
        sharepoint_manager = SharePointManager()
        print("📋 Testing access token...")
        token = sharepoint_manager.get_graph_access_token()

        if not token:
            return {"test_result": {"status": "error", "message": "Failed to get access token", "step": "token"}}

        print("✅ Access token obtained")

        # Test 2: Site access
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        site_url = f"https://graph.microsoft.com/v1.0/sites/{sharepoint_manager.graph_config['site_url']}"
        print(f"🌐 Testing site access: {site_url}")

        import requests
        site_response = requests.get(site_url, headers=headers)
        print(f"📊 Site response: {site_response.status_code}")

        if site_response.status_code != 200:
            return {
                "test_result": {
                    "status": "error",
                    "message": f"Site access failed: {site_response.status_code} - {site_response.text[:200]}",
                    "step": "site_access",
                    "site_url": site_url
                }
            }

        print("✅ Site access successful")
        return {"test_result": {"status": "success", "message": "SharePoint connection working", "site_data": site_response.json()}}

    except Exception as e:
        print(f"❌ SharePoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"test_result": {"status": "error", "message": str(e), "step": "exception"}}

@app.get("/api/files/{folder_path:path}")
async def get_folder_contents(folder_path: str):
    """Get contents of a specific SharePoint folder"""
    try:
        sharepoint_manager = SharePointManager()
        result = sharepoint_manager.get_folder_contents_by_path(folder_path)

        if result['status'] == 'success':
            # Transform SharePoint files to match frontend expectations
            transformed_files = []

            for item in result['files']:
                if item['type'] == 'folder':
                    # Add folder
                    folder_info = {
                        "filename": item['name'],
                        "file_size": 0,
                        "modified_at": item['modified'],
                        "type": "folder",
                        "web_url": item['web_url'],
                        "path": f"{folder_path}/{item['name']}" if folder_path else item['name'],
                        "id": item['id']
                    }
                    transformed_files.append(folder_info)
                else:
                    # Add file
                    file_info = {
                        "filename": item['name'],
                        "file_size": item['size'],
                        "modified_at": item['modified'],
                        "type": item['name'].split('.')[-1].lower() if '.' in item['name'] else 'file',
                        "web_url": item['web_url'],
                        "download_url": item['download_url'],
                        "path": f"{folder_path}/{item['name']}" if folder_path else item['name'],
                        "id": item['id']
                    }
                    transformed_files.append(file_info)

            return {
                "files": transformed_files,
                "status": "success",
                "message": f"Retrieved {len(transformed_files)} items from folder: {folder_path}",
                "current_path": folder_path
            }
        else:
            return {
                "files": [],
                "status": "error",
                "message": result.get('message', 'Failed to get folder contents'),
                "current_path": folder_path
            }

    except Exception as e:
        print(f"Error getting folder contents: {e}")
        return {
            "files": [],
            "status": "error",
            "message": str(e),
            "current_path": folder_path
        }

@app.get("/api/files")
async def list_files():
    """Get files from SharePoint"""
    try:
        sharepoint_manager = SharePointManager()
        result = sharepoint_manager.get_sharepoint_files()

        if result['status'] == 'success':
            # Transform SharePoint files to match frontend expectations
            transformed_files = []

            def process_items(items, parent_path=""):
                for item in items:
                    if item['type'] == 'folder':
                        # Add folder
                        folder_info = {
                            "filename": item['name'],
                            "file_size": 0,
                            "modified_at": item['modified'],
                            "type": "folder",
                            "web_url": item['web_url'],
                            "path": f"{parent_path}/{item['name']}" if parent_path else item['name'],
                            "children": []
                        }

                        # Process folder contents
                        if 'children' in item:
                            folder_info['children'] = []
                            for child in item['children']:
                                child_info = {
                                    "filename": child['name'],
                                    "file_size": child['size'],
                                    "modified_at": child['modified'],
                                    "type": child['name'].split('.')[-1].lower() if '.' in child['name'] else 'file',
                                    "web_url": child['web_url'],
                                    "download_url": child['download_url'],
                                    "path": f"{folder_info['path']}/{child['name']}"
                                }
                                folder_info['children'].append(child_info)

                        transformed_files.append(folder_info)
                    else:
                        # Add file
                        file_info = {
                            "filename": item['name'],
                            "file_size": item['size'],
                            "modified_at": item['modified'],
                            "type": item['name'].split('.')[-1].lower() if '.' in item['name'] else 'file',
                            "web_url": item['web_url'],
                            "download_url": item['download_url'],
                            "path": f"{parent_path}/{item['name']}" if parent_path else item['name']
                        }
                        transformed_files.append(file_info)

            process_items(result['files'])

            return {
                "files": transformed_files,
                "status": "success",
                "message": f"Retrieved {len(transformed_files)} items from SharePoint"
            }
        else:
            # Fallback to local files if SharePoint fails
            print(f"SharePoint failed: {result.get('message', 'Unknown error')}")
            print("Falling back to local file system...")
            return await get_local_files()

    except Exception as e:
        print(f"Error in list_files: {e}")
        print("Falling back to local file system...")
        return await get_local_files()

async def get_local_files():
    """Fallback to local file system"""
    try:
        import os
        import datetime
        files_dir = os.path.join(os.path.dirname(__file__), "..", "ba-server", "generated_files")
        files = []

        if os.path.exists(files_dir):
            for filename in os.listdir(files_dir):
                file_path = os.path.join(files_dir, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    file_type = filename.split('.')[-1] if '.' in filename else "unknown"

                    files.append({
                        "filename": filename,
                        "file_size": file_size,
                        "modified_at": file_modified.isoformat(),
                        "type": file_type
                    })

        return {"files": files, "status": "success", "message": "Using local files (SharePoint unavailable)"}
    except Exception as e:
        print(f"Local file listing failed: {e}")
        return {"files": [], "status": "error", "message": str(e)}

@app.get("/api/files/{filename}")
async def download_file(filename: str):
    from fastapi.responses import FileResponse
    import os
    
    # Path to generated files directory
    file_path = os.path.join(os.path.dirname(__file__), "..", "ba-server", "generated_files", filename)
    
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)