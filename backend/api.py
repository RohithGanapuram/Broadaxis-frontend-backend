

import asyncio
import json
import os
import time
import base64
import logging
import uuid
import bcrypt
import jwt
from typing import Dict, List, Optional
import re
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, UploadFile, WebSocket, Request, Depends, HTTPException, status, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ValidationError as PydanticValidationError
from dotenv import load_dotenv
# NEW imports for local-upload flow
import io
import os
import uuid
import re
from typing import Dict, List, Tuple
try:
    import PyPDF2  # we use PyPDF2 because it's already in requirements
except Exception:
    PyPDF2 = None
try:
    import docx  # python-docx
except Exception:
    docx = None


# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Setup Sentry for error tracking (with error handling)
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.redis import RedisIntegration

    # Initialize Sentry
    sentry_sdk.init(
        dsn="https://70c65fd67ca8d2cad3f6852505d6ad95@o4509991991705600.ingest.us.sentry.io/4509992009334784",
        integrations=[
            FastApiIntegration(),
            RedisIntegration(),
        ],
        # Performance monitoring
        traces_sample_rate=0.1,  # 10% of transactions
        # Error sampling
        sample_rate=1.0,  # 100% of errors
        # Add data like request headers and IP for users
        send_default_pii=True,
        # Environment
        environment=os.getenv('ENVIRONMENT', 'production'),
    )
    print("✅ Sentry initialized successfully")
except Exception as e:
    print(f"⚠️ Sentry initialization failed: {e}")
    print("🔄 Continuing without Sentry...")

from error_handler import (
    BroadAxisError, ValidationError, ExternalAPIError, FileOperationError, error_handler
)

# Import MCP interface
from mcp_interface import mcp_interface, run_mcp_query
from token_manager import token_manager
from document_prioritizer import document_prioritizer

# Import WebSocket functionality
from websocket_api import websocket_chat

# Import API routers
from email_api import email_router
from sharepoint_api import sharepoint_router, SharePointManager

# Import session manager (optional for now)
try:
    from session_manager import session_manager
    SESSION_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Session manager not available: {e}")
    SESSION_MANAGER_AVAILABLE = False
    session_manager = None

def safe_extract_text_content(content, index=0):
    """Safely extract text content from MCP tool result content, handling TextBlock objects"""
    if not content or len(content) <= index:
        return None
    
    item = content[index]
    if hasattr(item, 'text'):
        return item.text
    elif isinstance(item, dict) and 'text' in item:
        return item['text']
    else:
        return str(item)

app = FastAPI(title="BroadAxis API", version="1.0.0")

# CORS Configuration - Environment-based with security
CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS")
if CORS_ORIGINS_ENV:
    # Production: Use specific origins from environment
    CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_ENV.split(",") if origin.strip()]
    print(f"🔒 Production CORS: {CORS_ORIGINS}")
else:
    # Development: Allow localhost origins only
    CORS_ORIGINS = [
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173"
    ]
    print(f"🛠️ Development CORS: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

# CORS is handled by middleware above

# OPTIONS requests are handled by CORS middleware

# Include API routers
app.include_router(email_router)
app.include_router(sharepoint_router)



# ===== Local Upload Store (per-process) =====
# Shape: { session_id: { doc_id: { filename, pages, chunks, preview } } }
UPLOAD_STORE: Dict[str, Dict[str, Dict]] = {}

MAX_CHARS_PER_CHUNK = 1500   # ~400–500 tokens each
PREVIEW_CHARS = 1200         # what we return to UI on upload
MAX_RETURN_CHUNKS = 10       # cap search results

def _safe_text(s: str) -> str:
    return (s or "").replace("\x00", " ").strip()

def _pdf_to_pages(file_bytes: bytes) -> List[str]:
    if not PyPDF2:
        raise HTTPException(500, "PyPDF2 not installed on server")
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages: List[str] = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        pages.append(_safe_text(t))
    return pages

def _docx_to_text(file_bytes: bytes) -> str:
    if not docx:
        raise HTTPException(500, "python-docx not installed on server")
    f = io.BytesIO(file_bytes)
    d = docx.Document(f)
    return "\n".join(_safe_text(p.text) for p in d.paragraphs)

def _split_into_chunks(pages: List[str], max_chars: int = 1200, overlap_chars: int = 200) -> List[Dict]:
    """
    Sliding window chunking: smaller chunks with tail overlap so content that
    straddles boundaries (e.g., contact blocks, definitions) is still retrievable.
    """
    chunks: List[Dict] = []
    buf, buf_len, start_idx = [], 0, 0
 
    def flush(end_i):
        if not buf: return
        text = "\n".join(buf)
        chunks.append({
            "page_start": start_idx + 1,
            "page_end": end_i + 1,
            "text": text
        })
 
    for i, page in enumerate(pages):
        t = _safe_text(page)
        if not t:
            continue
        if buf and buf_len + len(t) + 1 > max_chars:
            flush(i - 1)
            # keep an overlapping tail from previous chunk
            if chunks:
                tail = chunks[-1]["text"][-overlap_chars:]
                buf = [tail, t] if tail else [t]
                buf_len = len("\n".join(buf))
                start_idx = i - 1
            else:
                buf, buf_len, start_idx = [t], len(t), i
        else:
            if not buf:
                start_idx = i
            buf.append(t)
            buf_len += len(t) + 1
 
    flush(len(pages) - 1)
    return chunks
 
 
_word = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
def _tokenize(s: str) -> List[str]:
    return [w.lower() for w in _word.findall(s or "")]
 
import math, re
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"\b(?:\+?\d{1,2}\s*)?(?:\(?\d{3}\)?[\s.-]*)?\d{3}[\s.-]?\d{4}\b")
 
def _score_chunks(query: str, chunks: List[Dict]) -> List[Tuple[float, Dict]]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return [(0.0, c) for c in chunks]
 
    # corpus stats
    N = len(chunks)
    chunk_tokens = []
    df = {}
    for c in chunks:
        toks = _tokenize(c["text"])
        chunk_tokens.append(toks)
        seen = set()
        for t in toks:
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
 
    idf = {t: math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0) for t, df_t in df.items()}
    k1, b = 1.5, 0.75
    qset = set(q_tokens)
    phrase = " ".join(q_tokens).strip()
 
    scored = []
    avgdl = max(sum(len(ct) for ct in chunk_tokens) / N, 1)
    for i, c in enumerate(chunks):
        toks = chunk_tokens[i]
        dl = max(len(toks), 1)
        tf = {}
        for t in toks:
            if t in qset:
                tf[t] = tf.get(t, 0) + 1
 
        score = 0.0
        for t, f in tf.items():
            score += idf.get(t, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avgdl)))
 
        if phrase and phrase in c["text"].lower():
            score += 1.5  # small phrase bonus
 
        if c["page_start"] <= 2:
            score += 0.8  # front-matter bias
        if EMAIL.search(c["text"]) or PHONE.search(c["text"]):
            score += 0.8  # likely contact/info blocks
 
        scored.append((float(score), c))
 
    scored.sort(key=lambda x: (-x[0], x[1]["page_start"]))
    return scored
 
 

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
    model: str = "claude-sonnet-4-5-20250929"



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

class RFPProcessingRequest(BaseModel):
    folder_path: str
    session_id: str = "default"

# Authentication Models
class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: str
    last_login: Optional[str] = None

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class TokenData(BaseModel):
    user_id: Optional[str] = None

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str
    
class LocalSearchRequest(BaseModel):
    session_id: str
    query: str
    k: Optional[int] = 8
class SearchBody(BaseModel):
    session_id: str
    query: str
    k: int = 8

# Trading access allowlist (comma-separated user emails in env), fallback to predefined emails
_env_allow = os.getenv("TRADING_ALLOWED_EMAILS")
if _env_allow:
    TRADING_ALLOWED_EMAILS = set((_env_allow.strip() or "").split(","))
else:
    print("⚠️ set trading allowed emails in the environment")

# Session file storage (keeping for backward compatibility)
session_files = {}

# Authentication Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    # Development fallback - WARNING: Change this in production!
    SECRET_KEY = "dev-secret-key-change-in-production"
    print("⚠️ WARNING: Using development JWT secret key. Set JWT_SECRET_KEY in .env for production!")
else:
    print("🔒 Production JWT secret key configured")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Authentication Helper Functions
def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify a JWT token and return user_id"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except jwt.PyJWTError:
        return None

# Authentication Middlewar
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """Authentication dependency to get current user from JWT token"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session management not available"
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        token = credentials.credentials
        
        # Verify token and get user_id
        user_id = verify_token(token)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if session exists in Redis
        session_data = await session_manager.redis.get(f"session:{token}")
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user data
        user_data_str = await session_manager.redis.get(f"user:{user_id}")
        if not user_data_str:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user = json.loads(user_data_str)
        
        return UserResponse(
            id=user["id"],
            name=user["name"],
            email=user["email"],
            created_at=user["created_at"],
            last_login=user.get("last_login")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

async def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[UserResponse]:
    """Optional authentication dependency - returns None if not authenticated"""
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None

# run_mcp_query is now imported from mcp_interface.py

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

@app.get("/api/access/trading")
async def trading_access(current_user: UserResponse = Depends(get_current_user)):
    """Return whether current user has trading planner access."""
    return {
        "trading_access": current_user.email in TRADING_ALLOWED_EMAILS,
        "status": "success"
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
        # Ensure we have an active connection
        await mcp_interface._ensure_connection()
        
        # Call document processing tool using persistent session
        result = await mcp_interface.session.call_tool(
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
        
        # Convert TextBlock objects to strings for JSON serialization
        message = "Document processed successfully"
        if result.content:
            if hasattr(result.content[0], 'text'):
                message = result.content[0].text
            elif isinstance(result.content[0], dict) and 'text' in result.content[0]:
                message = result.content[0]['text']
            else:
                message = str(result.content[0])
        
        return {
            "status": "success",
            "filename": file.filename,
            "size": len(file_content),
            "files_in_session": len(session_files[session_key]),
            "message": message
        }
        
    except BroadAxisError:
        raise
    except Exception as e:
        error_handler.log_error(e, {'filename': file.filename, 'operation': 'file_upload'})
        raise FileOperationError("Failed to process uploaded file", {'filename': file.filename})


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

# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication."""
    await websocket_chat(websocket)

# WebSocket endpoint for real-time RFP processing


@app.post("/api/chat")
async def chat_with_context(request: ChatRequest, session_id: str = None, current_user: UserResponse = Depends(get_current_user)):
    """Chat endpoint with Redis session management"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            # Fallback to basic chat without session management
            result = await run_mcp_query(
                request.query, 
                request.enabled_tools, 
                request.model, 
                session_id or "default"
            )
            return {
                "response": result["response"],
                "session_id": session_id or "default",
                "conversation_length": 1,
                "status": "success",
                "note": "Session management not available"
            }
        
        # Create new session if none provided
        if not session_id:
            session_id = await session_manager.create_session()
            print(f"🆕 Created new session: {session_id}")
        
        # Get conversation history
        conversation_history = await session_manager.get_conversation_context(session_id)
        
        # Add user message to history
        user_message = {
            "role": "user",
            "content": request.query,
            "timestamp": datetime.now().isoformat()
        }
        await session_manager.add_message(session_id, user_message)
        
        # Include conversation history in AI prompt
        context_prompt = ""
        if conversation_history:
            context_prompt = "Previous conversation:\n"
            for msg in conversation_history[-10:]:  # Last 10 messages
                context_prompt += f"{msg['role']}: {msg['content']}\n"
            context_prompt += "\nCurrent question: "
        
        # Process with AI (include context)
        result = await run_mcp_query(
            context_prompt + request.query, 
            request.enabled_tools, 
            request.model, 
            session_id
        )
        
        # Add AI response to history
        ai_message = {
            "role": "assistant", 
            "content": result["response"],
            "timestamp": datetime.now().isoformat()
        }
        await session_manager.add_message(session_id, ai_message)
        
        return {
            "response": result["response"],
            "session_id": session_id,
            "conversation_length": len(conversation_history) + 2,
            "status": "success"
        }
        
    except Exception as e:
        error_handler.log_error(e, {'operation': 'chat_with_context', 'session_id': session_id})
        raise

# ---------------- TRADING PLANNER (restricted) ----------------
class TradingChatRequest(BaseModel):
    query: str
    model: str = "claude-sonnet-4-5-20250929"
    session_id: Optional[str] = None

TRADING_SYSTEM_PROMPT = """
You are BroadAxis Trading Planner - an advanced AI trading assistant with smart tool selection capabilities.

## 🎯 Core Mission:
Provide comprehensive trading analysis with intelligent tool selection based on query type.

## 📊 Available Tools & Smart Selection:
- **batch_earnings_analysis**: Yahoo Finance data for options, prices, expected moves, 200% rule calculations
- **web_search_tool**: Latest news, market updates, breaking developments, market sentiment

## 🔍 Smart Tool Selection Strategy:

### **📈 Earnings Analysis Queries** (Use batch_earnings_analysis only):
- "Analyze earnings for AVAV, ORCL, SNPS"
- "Weekly earnings planner"
- "Options analysis for earnings"
- "200% rule calculations"
- **Tool**: `batch_earnings_analysis` only (cost-effective, focused)

### **📰 General Trading Queries** (Use both tools):
- "What's happening with NVDA today?"
- "Latest news on Tesla"
- "Market analysis for tech stocks"
- "Current market conditions"
- **Tools**: `batch_earnings_analysis` + `web_search_tool` (comprehensive)

### **🌐 Market News Queries** (Use web_search_tool only):
- "Latest market news"
- "Breaking developments"
- "Market sentiment today"
- **Tool**: `web_search_tool` only (latest updates)

## 📊 Earnings Analysis Format (When Applicable):
Break down by day (Mon, Tue, Wed, Thu) with sections:
- Running Man (post earnings same-day follow-up)
- PRE (earnings before open or after close, next-day open)  
- POST (after earnings release, setups for post IV-crush plays)
- Closing (AMC earnings of that day)

## 📊 Table Structure (For Earnings Analysis):
| Symbol | Expiry | Current | Call K | Call Px | Put K | Put Px | EM (C+P) | 200% Call | 200% Put | Bias (60/40) | Confidence | Hist Move |

## ⚠️ Critical Rules:
- **Smart tool selection** - choose appropriate tools based on query type
- **Cost optimization** - use minimal tools needed for the query
- **Professional formatting** - clean, readable output
- **Accurate data** - always use real-time information when available
- **Bold PASS/FAIL entries** (for earnings analysis)
- **NO extra commentary in tables** (for earnings analysis)

## 🎯 Tool Usage Guidelines:
- **Earnings queries**: Use batch_earnings_analysis only
- **General trading**: Use both tools for comprehensive analysis
- **News queries**: Use web_search_tool only
- **Always prioritize** the most relevant tool for the query type

Remember: Smart tool selection ensures optimal cost-effectiveness while providing comprehensive analysis when needed.
"""

def _require_trading_access(user: UserResponse):
    # Temporarily allow all authenticated users to access trading planner
    # TODO: Add proper email allowlist management
    if user.email not in TRADING_ALLOWED_EMAILS:
        # Log the user email for debugging
        logger.info(f"User {user.email} requesting trading access - adding to allowed list")
        # Add user to allowed list for this session
        TRADING_ALLOWED_EMAILS.add(user.email)
        # For now, allow access instead of raising 403
        # raise HTTPException(status_code=403, detail="Not authorized for trading planner")

@app.post("/api/trading/session/create")
async def trading_create_session(current_user: UserResponse = Depends(get_current_user)):
    _require_trading_access(current_user)
    if not session_manager.redis:
        await session_manager.connect()
    session_id = await session_manager.create_session(user_id=current_user.id)
    await session_manager.redis.sadd(f"user_trading_sessions:{current_user.id}", session_id)
    return {"session_id": session_id, "status": "success"}

@app.get("/api/trading/sessions")
async def trading_list_sessions(current_user: UserResponse = Depends(get_current_user)):
    _require_trading_access(current_user)
    if not session_manager.redis:
        await session_manager.connect()
    session_ids = await session_manager.redis.smembers(f"user_trading_sessions:{current_user.id}")
    sessions = []
    for sid in session_ids:
        data = await session_manager.get_session(sid)
        if data:
            sessions.append({
                "id": sid,
                "title": data.get("title", "Trading Chat"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "message_count": len(data.get("messages", []))
            })
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {"sessions": sessions, "status": "success"}

@app.get("/api/trading/session/{session_id}")
async def trading_get_session(session_id: str, current_user: UserResponse = Depends(get_current_user)):
    _require_trading_access(current_user)
    data = await session_manager.get_session(session_id)
    if not data:
        return {"status": "error", "error": "Session not found"}
    return {
        "status": "success",
        "session_id": session_id,
        "messages": data.get("messages", []),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at")
    }

@app.delete("/api/trading/session/{session_id}")
async def trading_delete_session(session_id: str, current_user: UserResponse = Depends(get_current_user)):
    _require_trading_access(current_user)
    if not session_manager.redis:
        await session_manager.connect()
    await session_manager.delete_session(session_id)
    await session_manager.redis.srem(f"user_trading_sessions:{current_user.id}", session_id)
    return {"status": "success"}

@app.post("/api/trading/chat")
async def trading_chat(request: TradingChatRequest, current_user: UserResponse = Depends(get_current_user)):
    _require_trading_access(current_user)
    try:
        session_id = request.session_id or await session_manager.create_session(user_id=current_user.id)
        # Add user message
        await session_manager.add_message(session_id, {
            "role": "user",
            "content": request.query,
            "timestamp": datetime.now().isoformat()
        })
        # Enable smart tool selection for comprehensive trading analysis using Claude Sonnet 3.7
        result = await run_mcp_query(
            query=request.query,
            enabled_tools=["batch_earnings_analysis", "web_search_tool"],  # Smart selection based on query type
            model="claude-sonnet-4-5-20250929",  # Use Sonnet 3.7 for cost efficiency
            session_id=session_id,
            system_prompt=TRADING_SYSTEM_PROMPT
        )
        # Store assistant message
        await session_manager.add_message(session_id, {
            "role": "assistant",
            "content": result.get("response", ""),
            "timestamp": datetime.now().isoformat()
        })
        return {
            "status": "success",
            "response": result.get("response", ""),
            "session_id": session_id
        }
    except Exception as e:
        error_handler.log_error(e, {"operation": "trading_chat"})
        return JSONResponse(status_code=500, content={"error": "Trading chat failed"})


@app.post("/api/session/create")
async def create_session(current_user: UserResponse = Depends(get_current_user)):
    """Create a new session"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "session_id": None,
                "error": "Session management not available",
                "status": "error"
            }
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Create session with user ID
        session_id = await session_manager.create_session(user_id=current_user.id)
        
        # Store user-session mapping
        await session_manager.redis.sadd(f"user_sessions:{current_user.id}", session_id)
        
        return {
            "session_id": session_id,
            "status": "success"
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'create_session'})
        raise

@app.get("/api/user/sessions")
async def get_user_sessions(current_user: UserResponse = Depends(get_current_user)):
    """Get all sessions for the current user"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "sessions": [],
                "error": "Session management not available",
                "status": "error"
            }
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Get all session IDs for this user
        session_ids = await session_manager.redis.smembers(f"user_sessions:{current_user.id}")
        
        sessions = []
        for session_id in session_ids:
            session_data = await session_manager.get_session(session_id)
            if session_data:
                sessions.append({
                    "id": session_id,
                    "title": session_data.get("title", f"Chat {session_id[:8]}"),
                    "created_at": session_data.get("created_at"),
                    "updated_at": session_data.get("updated_at"),
                    "message_count": len(session_data.get("messages", []))
                })
        
        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return {
            "sessions": sessions,
            "status": "success"
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_user_sessions'})
        return {
            "sessions": [],
            "error": str(e),
            "status": "error"
        }

@app.get("/api/admin/active-users")
async def get_active_users(current_user: UserResponse = Depends(get_current_user)):
    """Get all active users with email addresses and organized session data (admin endpoint)"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "active_users": [],
                "error": "Session management not available",
                "status": "error"
            }
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Get all user session keys
        user_session_keys = await session_manager.redis.keys("user_sessions:*")
        user_trading_session_keys = await session_manager.redis.keys("user_trading_sessions:*")
        
        # Collect all unique user IDs
        all_user_ids = set()
        for key in user_session_keys:
            user_id = key.replace("user_sessions:", "")
            all_user_ids.add(user_id)
        for key in user_trading_session_keys:
            user_id = key.replace("user_trading_sessions:", "")
            all_user_ids.add(user_id)
        
        active_users = []
        
        # Process each user
        for user_id in all_user_ids:
            # Get user email from Redis
            user_email = await session_manager.redis.get(f"user:email:{user_id}")
            if not user_email:
                # Try to get email from user data
                user_data_str = await session_manager.redis.get(f"user:{user_id}")
                if user_data_str:
                    user_data = json.loads(user_data_str)
                    user_email = user_data.get("email", f"Unknown User ({user_id[:8]}...)")
                else:
                    user_email = f"Unknown User ({user_id[:8]}...)"
            
            # Get user name
            user_name = "Unknown"
            user_data_str = await session_manager.redis.get(f"user:{user_id}")
            if user_data_str:
                user_data = json.loads(user_data_str)
                user_name = user_data.get("name", "Unknown")
            
            # Collect all sessions for this user
            rfp_sessions = []
            trading_sessions = []
            last_activity = None
            
            # Get regular (RFP) sessions
            rfp_session_ids = await session_manager.redis.smembers(f"user_sessions:{user_id}")
            for session_id in rfp_session_ids:
                session_data = await session_manager.get_session(session_id)
                if session_data:
                    session_info = {
                        "id": session_id,
                        "title": session_data.get("title", "RFP Chat"),
                        "created_at": session_data.get("created_at"),
                        "updated_at": session_data.get("updated_at"),
                        "message_count": len(session_data.get("messages", [])),
                        "activity_status": _get_activity_status(session_data.get("updated_at"))
                    }
                    rfp_sessions.append(session_info)
                    
                    # Track most recent activity
                    session_updated = session_data.get("updated_at")
                    if session_updated and (not last_activity or session_updated > last_activity):
                        last_activity = session_updated
            
            # Get trading sessions
            trading_session_ids = await session_manager.redis.smembers(f"user_trading_sessions:{user_id}")
            for session_id in trading_session_ids:
                session_data = await session_manager.get_session(session_id)
                if session_data:
                    session_info = {
                        "id": session_id,
                        "title": session_data.get("title", "Trading Planner"),
                        "created_at": session_data.get("created_at"),
                        "updated_at": session_data.get("updated_at"),
                        "message_count": len(session_data.get("messages", [])),
                        "activity_status": _get_activity_status(session_data.get("updated_at"))
                    }
                    trading_sessions.append(session_info)
                    
                    # Track most recent activity
                    session_updated = session_data.get("updated_at")
                    if session_updated and (not last_activity or session_updated > last_activity):
                        last_activity = session_updated
            
            # Sort sessions by updated_at (most recent first)
            rfp_sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            trading_sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            
            # Only include users who have sessions
            if rfp_sessions or trading_sessions:
                user_info = {
                    "user_id": user_id,
                    "user_name": user_name,
                    "user_email": user_email,
                    "last_activity": last_activity,
                    "activity_status": _get_activity_status(last_activity),
                    "total_sessions": len(rfp_sessions) + len(trading_sessions),
                    "rfp_chat": {
                        "session_count": len(rfp_sessions),
                        "sessions": rfp_sessions
                    },
                    "trading_planner": {
                        "session_count": len(trading_sessions),
                        "sessions": trading_sessions
                    }
                }
                active_users.append(user_info)
        
        # Sort users by last activity (most recent first)
        active_users.sort(key=lambda x: x.get("last_activity", ""), reverse=True)
        
        # Add summary statistics
        total_rfp_sessions = sum(user["rfp_chat"]["session_count"] for user in active_users)
        total_trading_sessions = sum(user["trading_planner"]["session_count"] for user in active_users)
        
        return {
            "active_users": active_users,
            "summary": {
                "total_users": len(active_users),
                "total_rfp_sessions": total_rfp_sessions,
                "total_trading_sessions": total_trading_sessions,
                "total_sessions": total_rfp_sessions + total_trading_sessions
            },
            "status": "success"
        }
        
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_active_users'})
        return {
            "active_users": [],
            "error": str(e),
            "status": "error"
        }

def _get_activity_status(last_activity: str) -> str:
    """Determine activity status based on last activity timestamp"""
    if not last_activity:
        return "Never Active"
    
    try:
        from datetime import datetime, timezone
        last_activity_dt = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        time_diff = now - last_activity_dt
        
        if time_diff.total_seconds() < 3600:  # Less than 1 hour
            return "Active Now"
        elif time_diff.total_seconds() < 86400:  # Less than 24 hours
            return "Active Today"
        elif time_diff.total_seconds() < 604800:  # Less than 7 days
            return "Active This Week"
        else:
            return "Inactive"
    except:
        return "Unknown"

@app.post("/api/admin/clear-document-cache")
async def clear_document_cache(
    document_path: str = None, 
    current_user: UserResponse = Depends(get_current_user)
):
    """Clear document cache for consistent results (admin endpoint)"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "status": "error",
                "message": "Session management not available"
            }
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        await session_manager.clear_document_cache(document_path)
        
        return {
            "status": "success",
            "message": f"Document cache cleared for: {document_path or 'all documents'}"
        }
        
    except Exception as e:
        error_handler.log_error(e, {'operation': 'clear_document_cache'})
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/api/admin/clear-go-no-go-cache")
async def clear_go_no_go_cache(current_user: UserResponse = Depends(get_current_user)):
    """Clear Go/No-Go analysis cache for fresh analysis (admin endpoint)"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "status": "error",
                "message": "Session management not available"
            }
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Clear all Go/No-Go analysis caches
        pattern = "go_no_go_analysis:*"
        keys = await session_manager.redis.keys(pattern)
        if keys:
            await session_manager.redis.delete(*keys)
            print(f"🗑️ Cleared {len(keys)} Go/No-Go analysis caches")
            return {
                "status": "success",
                "message": f"Cleared {len(keys)} Go/No-Go analysis caches"
            }
        else:
            print("ℹ️ No Go/No-Go analysis caches found to clear")
            return {
                "status": "success",
                "message": "No Go/No-Go analysis caches found to clear"
            }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'clear_go_no_go_cache'})
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/admin/redis-keys")
async def get_redis_keys(current_user: UserResponse = Depends(get_current_user)):
    """Get all Redis keys for debugging (admin endpoint)"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "keys": [],
                "error": "Session management not available",
                "status": "error"
            }
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Get all keys
        all_keys = await session_manager.redis.keys("*")
        
        # Categorize keys
        session_keys = [k for k in all_keys if k.startswith("session:")]
        user_session_keys = [k for k in all_keys if k.startswith("user_sessions:")]
        user_trading_keys = [k for k in all_keys if k.startswith("user_trading_sessions:")]
        other_keys = [k for k in all_keys if not any(k.startswith(prefix) for prefix in ["session:", "user_sessions:", "user_trading_sessions:"])]
        
        return {
            "total_keys": len(all_keys),
            "session_keys": {
                "count": len(session_keys),
                "keys": session_keys[:10]  # Show first 10 to avoid overwhelming response
            },
            "user_session_keys": {
                "count": len(user_session_keys),
                "keys": user_session_keys
            },
            "user_trading_keys": {
                "count": len(user_trading_keys),
                "keys": user_trading_keys
            },
            "other_keys": {
                "count": len(other_keys),
                "keys": other_keys[:10]  # Show first 10
            },
            "status": "success"
        }
        
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_redis_keys'})
        return {
            "keys": [],
            "error": str(e),
            "status": "error"
        }

@app.delete("/api/session/{session_id}")
async def delete_session_api(session_id: str, current_user: UserResponse = Depends(get_current_user)):
    """Delete a session and remove it from the user's session set."""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        # Delete session data
        await session_manager.delete_session(session_id)
        # Remove mapping
        await session_manager.redis.srem(f"user_sessions:{current_user.id}", session_id)
        return {"status": "success", "deleted_session": session_id}
    except Exception as e:
        error_handler.log_error(e, {"operation": "delete_session", "session_id": session_id})
        return JSONResponse(status_code=500, content={"error": "Failed to delete session"})

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information and conversation history"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "session_id": session_id,
                "error": "Session management not available",
                "status": "error"
            }
        
        session = await session_manager.get_session(session_id)
        if session:
            return {
                "session_id": session_id,
                "messages": session.get("messages", []),
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at"),
                "message_count": len(session.get("messages", [])),
                "status": "success"
            }
        else:
            return {
                "session_id": session_id,
                "error": "Session not found",
                "status": "error"
            }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_session_info', 'session_id': session_id})
        raise


@app.get("/api/redis/status")
async def get_redis_status():
    """Get Redis connection and storage status"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "status": "not_available",
                "error": "Session management not available"
            }
        
        usage = await session_manager.get_storage_usage()
        
        # Test creating a session
        test_session_id = await session_manager.create_session()
        test_session = await session_manager.get_session(test_session_id)
        await session_manager.delete_session(test_session_id)
        
        return {
            "status": "connected",
            "storage_usage": usage,
            "usage_percentage": (usage['used_memory'] / usage['max_memory']) * 100 if usage['max_memory'] > 0 else 0,
            "test_session_created": test_session_id,
            "test_session_retrieved": test_session is not None
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_redis_status'})
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/token-status")
async def get_token_status():
    """Get current token budget and usage status"""
    try:
        budget_status = token_manager.get_budget_status()
        return {
            "status": "success",
            "budgets": budget_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_token_status'})
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/token-usage/{session_id}")
async def get_token_usage(session_id: str):
    """Get token usage statistics for a specific session"""
    try:
        usage_stats = token_manager.get_usage_stats(session_id)
        return {
            "status": "success",
            "usage": usage_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_token_usage', 'session_id': session_id})
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/token-usage")
async def get_all_token_usage():
    """Get overall token usage statistics"""
    try:
        usage_stats = token_manager.get_usage_stats()
        return {
            "status": "success",
            "usage": usage_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_all_token_usage'})
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/token-usage/query/{request_id}")
async def get_query_token_usage(request_id: str, current_user: UserResponse = Depends(get_current_user)):
    """Get token usage for a specific query/request"""
    try:
        # Find the specific request in usage history
        for usage in token_manager.usage_history:
            if usage.request_id == request_id:
                return {
                    "status": "success",
                    "query_usage": {
                        "request_id": usage.request_id,
                        "model": usage.model,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "task_type": usage.task_type,
                        "session_id": usage.session_id,
                        "timestamp": usage.timestamp.isoformat()
                    }
                }
        
        return {
            "status": "error",
            "error": "Request not found"
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_query_token_usage'})
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/token-usage/user/{user_id}")
async def get_user_token_usage(user_id: str, current_user: UserResponse = Depends(get_current_user)):
    """Get total token usage for a specific user across all their sessions"""
    try:
        # Get all sessions for this user
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "status": "error",
                "error": "Session management not available"
            }
        
        # Get user sessions from Redis
        if not session_manager.redis:
            await session_manager.connect()
        
        # Find all sessions for this user
        user_sessions = []
        pattern = "session:*"
        keys = await session_manager.redis.keys(pattern)
        
        for key in keys:
            session_data = await session_manager.redis.get(key)
            if session_data:
                session = json.loads(session_data)
                if session.get("user_id") == user_id:
                    session_id = key.replace("session:", "")
                    user_sessions.append(session_id)
        
        # Calculate total usage across all user sessions
        total_tokens = 0
        total_requests = 0
        model_breakdown = {}
        session_breakdown = {}
        
        for session_id in user_sessions:
            session_usage = token_manager.get_usage_stats(session_id)
            total_tokens += session_usage.get("total_tokens", 0)
            total_requests += session_usage.get("total_requests", 0)
            
            # Add to session breakdown
            session_breakdown[session_id] = {
                "tokens": session_usage.get("total_tokens", 0),
                "requests": session_usage.get("total_requests", 0)
            }
            
            # Add to model breakdown
            for model, stats in session_usage.get("models", {}).items():
                if model not in model_breakdown:
                    model_breakdown[model] = {"tokens": 0, "requests": 0}
                model_breakdown[model]["tokens"] += stats["tokens"]
                model_breakdown[model]["requests"] += stats["requests"]
        
        return {
            "status": "success",
            "user_usage": {
                "user_id": user_id,
                "total_tokens": total_tokens,
                "total_requests": total_requests,
                "model_breakdown": model_breakdown,
                "session_breakdown": session_breakdown,
                "total_sessions": len(user_sessions)
            }
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_user_token_usage'})
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/debug/document-prioritization")
async def debug_document_prioritization(folder_path: str, current_user: UserResponse = Depends(get_current_user)):
    """Debug endpoint to see how documents are being prioritized"""
    try:
        # List files in the folder
        list_result = await mcp_interface.session.call_tool(
            "sharepoint_list_files",
            arguments={"path": folder_path, "max_results": 100, "recursive": True}
        )
        
        if not list_result.content:
            return {
                "status": "error",
                "message": f"No documents found in folder: {folder_path}"
            }
        
        # Parse the list result
        content_text = safe_extract_text_content(list_result.content)
        if not content_text:
            return {
                "status": "error",
                "message": "Failed to extract content from list result"
            }
        files_data = json.loads(content_text)
        if files_data.get("status") != "success":
            return {
                "status": "error",
                "message": f"Failed to list files: {files_data.get('error', 'Unknown error')}"
            }
        
        items = files_data.get("items", [])
        documents = [item for item in items if not item.get("is_folder", False)]
        
        # Prioritize documents
        prioritized_docs = document_prioritizer.prioritize_documents(documents)
        
        # Get recommendations
        recommendation = document_prioritizer.get_processing_recommendation(prioritized_docs)
        
        return {
            "status": "success",
            "folder_path": folder_path,
            "total_documents": len(documents),
            "prioritized_documents": [
                {
                    "filename": doc.filename,
                    "priority": doc.priority.value,
                    "confidence": doc.confidence_score,
                    "indicators": doc.key_indicators,
                    "document_type": doc.document_type.value,
                    "estimated_tokens": doc.estimated_tokens
                }
                for doc in prioritized_docs
            ],
            "recommendation": recommendation
        }
        
    except Exception as e:
        error_handler.log_error(e, {'operation': 'debug_document_prioritization'})
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/token-usage/detailed/{session_id}")
async def get_detailed_session_token_usage(session_id: str, current_user: UserResponse = Depends(get_current_user)):
    """Get detailed token usage for a session including individual queries"""
    try:
        # Get session usage stats
        session_stats = token_manager.get_usage_stats(session_id)
        
        # Get individual query details
        query_details = []
        for usage in token_manager.usage_history:
            if usage.session_id == session_id:
                query_details.append({
                    "request_id": usage.request_id,
                    "model": usage.model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "task_type": usage.task_type,
                    "timestamp": usage.timestamp.isoformat()
                })
        
        # Sort by timestamp (newest first)
        query_details.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "status": "success",
            "session_usage": {
                "session_id": session_id,
                "summary": session_stats,
                "query_details": query_details,
                "total_queries": len(query_details)
            }
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_detailed_session_token_usage'})
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/rfp-analyses/{session_id}")
async def get_rfp_analyses(session_id: str):
    """Get stored RFP analyses for a session"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "status": "error",
                "error": "Session management not available"
            }
        
        analyses = await session_manager.get_rfp_analyses(session_id)
        return {
            "status": "success",
            "analyses": analyses,
            "count": len(analyses),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_rfp_analyses', 'session_id': session_id})
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/document-summary/{session_id}")
async def get_document_summary(session_id: str, document_path: str):
    """Get stored document summary"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "status": "error",
                "error": "Session management not available"
            }
        
        summary = await session_manager.get_document_summary(session_id, document_path)
        if summary:
            return {
                "status": "success",
                "summary": summary,
                "document_path": document_path,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "not_found",
                "message": "Document summary not found",
                "document_path": document_path,
                "session_id": session_id
            }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_document_summary', 'session_id': session_id, 'document_path': document_path})
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/api/process-rfp-folder-intelligent")
async def process_rfp_folder_intelligent(request: RFPProcessingRequest, current_user: UserResponse = Depends(get_current_user)):
    """Intelligently process an RFP folder with document prioritization and token management"""
    
    def remove_duplicate_doc_header(analysis_text: str, filename: str) -> str:
        """Remove duplicate document headers from analysis text"""
        if not isinstance(analysis_text, str):
            return analysis_text
        
        # Remove various patterns of duplicate document headers
        patterns_to_remove = [
            f"📄 Document: {filename}",
            f"Document: {filename}",
            f"### 📄 **Document: {filename}**",
            f"### 📄 **{filename}**",
            f"**Document: {filename}**",
            f"**{filename}**"
        ]
        
        cleaned = analysis_text
        for pattern in patterns_to_remove:
            cleaned = cleaned.replace(pattern, "").strip()
        
        # Clean up any leftover markdown formatting
        cleaned = re.sub(r'\*{2,}', '', cleaned)  # Remove multiple asterisks
        cleaned = re.sub(r'#{3,}', '', cleaned)   # Remove multiple hash symbols
        
        # Remove any leading/trailing whitespace and empty lines
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned).strip()
        
        return cleaned
    
    def clean_markup(text: str) -> str:
        """Clean simple HTML-like tags and error messages the LLM may emit"""
        if not isinstance(text, str):
            return text
        
        # Remove HTML-like tags
        cleaned = text.replace('<result>', '').replace('</result>', '')
        cleaned = re.sub(r"<a[^>]*></a>\s*", "", cleaned)
        
        # Remove ALL error/apology/process messages that shouldn't be shown to users
        # These are comprehensive patterns to catch all variations
        error_phrases = [
            # Apologies and errors
            "Oops, my apologies",
            "Oops my apologies", 
            "my apologies",
            "I apologize",
            "Sorry",
            
            # File type confusion
            "The file type is DOCX, not PDF",
            "The file type is",
            "not PDF",
            "still an issue with the file type",
            
            # Process messages
            "Let me try extracting",
            "Let me try a different approach",
            "Let me extract the text",
            "Let me analyze",
            "Great, the file is available",
            "Hmm, still an issue",
            "Ah, got it",
            "Okay, got the file details",
            
            # Tool-calling commentary
            "instead:",
            "Let me try",
            "from the DOCX file",
            "from that:",
        ]
        
        # Remove each phrase (case-insensitive)
        for phrase in error_phrases:
            # Remove the phrase and everything up to the next period or colon
            cleaned = re.sub(rf"{re.escape(phrase)}[^📄]*?(?=\n\n|###|####|📄)", "", cleaned, flags=re.IGNORECASE)
        
        # Remove sentences that start with these patterns
        sentence_patterns = [
            r"Oops[^\.!?]*[\.!?]\s*",
            r"Let me[^\.!?]*[\.!?]\s*",
            r"Great,[^\.!?]*[\.!?]\s*",
            r"Hmm,[^\.!?]*[\.!?]\s*",
            r"Ah,[^\.!?]*[\.!?]\s*",
            r"Okay,[^\.!?]*[\.!?]\s*",
            r"The file type is[^\.!?]*[\.!?]\s*",
        ]
        
        for pattern in sentence_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Clean up multiple consecutive newlines (more than 2)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Remove any standalone colons at the start of lines
        cleaned = re.sub(r'^\s*:\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove empty sections (just a heading with no content)
        cleaned = re.sub(r'(####[^\n]+)\n+(?=####|###)', r'\1\n\n', cleaned)
        
        return cleaned.strip()
    
    folder_path = request.folder_path
    session_id = request.session_id
    try:
        # Ensure MCP connection
        await mcp_interface._ensure_connection()
        
        # Step 1: Recursively list all documents in the folder and subfolders
        print(f"🔍 Attempting to list files in folder: '{folder_path}'")
        list_result = await mcp_interface.session.call_tool(
            "sharepoint_list_files",
            arguments={"path": folder_path, "max_results": 100, "recursive": True}
        )
        
        print(f"📋 SharePoint list result: {list_result}")
        print(f"📋 Content: {list_result.content}")
        
        if not list_result.content:
            return {
                "status": "error",
                "message": f"No documents found in the specified folder: '{folder_path}'. Please check if the folder exists and contains files."
            }
        
        # Parse the list result
        try:
            content_text = safe_extract_text_content(list_result.content)
            if not content_text:
                return {
                    "status": "error",
                    "message": "Failed to extract content from list result"
                }
            print(f"📄 Raw content text: {content_text}")
            files_data = json.loads(content_text)
            print(f"📄 Parsed files data: {files_data}")
            
            if files_data.get("status") != "success":
                return {
                    "status": "error",
                    "message": f"Failed to list files: {files_data.get('error', 'Unknown error')}"
                }
            items = files_data.get("items", [])
            print(f"📄 Found {len(items)} items: {[item.get('name', 'unknown') for item in items]}")
            
            # Filter out folders and only keep actual files
            documents = [item for item in items if not item.get("is_folder", False)]
            folders = [item for item in items if item.get("is_folder", False)]
            
            print(f"📁 Found {len(folders)} folders: {[f.get('name') for f in folders]}")
            print(f"📄 Found {len(documents)} documents: {[d.get('name') for d in documents]}")
            
            # Check if we have a single archive file that needs to be extracted
            if len(documents) == 1 and len(folders) == 0:
                archive_item = documents[0]
                archive_name = archive_item.get("name", "")
                archive_path = archive_item.get("path", "")
                
                # Check if it's likely an archive file (ZIP, RAR, etc.)
                if any(ext in archive_name.lower() for ext in ['.zip', '.rar', '.7z', '.tar', '.gz']):
                    print(f"📦 Detected archive file: {archive_name}")
                    return {
                        "status": "error",
                        "message": f"Archive file detected: '{archive_name}'. Please extract the contents first or use a different folder path.",
                        "suggestion": f"Try using the path: '{archive_path}' or extract the archive to get individual documents."
                    }
                else:
                    # It's a single document file - this is perfectly fine to process
                    print(f"📄 Single document found: {archive_name} - proceeding with analysis")
                    # Continue processing - don't return an error
            
            # If we have folders but no documents, that's also an issue
            if len(documents) == 0 and len(folders) > 0:
                return {
                    "status": "error",
                    "message": f"Found {len(folders)} folders but no documents. The intelligent RFP processing needs actual document files.",
                    "suggestion": f"Folders found: {[f.get('name') for f in folders]}. Please ensure these folders contain document files."
                }
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return {
                "status": "error",
                "message": f"Invalid response from SharePoint list files: {str(e)}"
            }
        
        if not documents:
            return {
                "status": "error",
                "message": "No documents found in the specified folder"
            }
        
        # Step 2: Prioritize documents
        prioritized_docs = document_prioritizer.prioritize_documents(documents)
        
        # Step 3: Get processing recommendation
        recommendation = document_prioritizer.get_processing_recommendation(prioritized_docs)
        
        # Step 4: Process primary documents first (if any)
        primary_docs = document_prioritizer.get_primary_documents(prioritized_docs, max_count=10)  # Increased from 3 to 10
        secondary_docs = document_prioritizer.get_secondary_documents(prioritized_docs, max_count=5)  # Increased from 2 to 5
        
        processed_documents = []
        total_tokens_used = 0
        
        # Process primary documents
        for doc in primary_docs:
            try:
                # Check if we already have a summary for this document
                existing_summary = None
                if SESSION_MANAGER_AVAILABLE:
                    existing_summary = await session_manager.get_document_summary(session_id, doc.file_path)
                
                if existing_summary:
                    # Use cached summary
                    processed_documents.append({
                        "filename": doc.filename,
                        "priority": doc.priority.value,
                        "confidence": doc.confidence_score,
                        "analysis": existing_summary["summary"]["analysis"],
                        "tokens_used": 0,  # No tokens used for cached result
                        "model_used": "cached",
                        "cached": True
                    })
                    logger.info(f"Using cached analysis for {doc.filename}")
                else:
                    # Use comprehensive RFP analysis prompt following BroadAxis-AI format
                    analysis_prompt = f"""You are **BroadAxis-AI**, an intelligent assistant that analyzes procurement documents (RFP, RFQ, RFI) to help vendor teams quickly understand the opportunity and make informed pursuit decisions.

## 📄 **Document to Analyze:**
**Filename:** {doc.filename}

## 🔍 **Analysis Instructions:**
1. **For PRIMARY documents:** Use `extract_pdf_text(path="{doc.file_path}", pages="all", max_pages=200)` to read the ENTIRE document
2. **For SECONDARY documents:** Use `extract_pdf_text(path="{doc.file_path}", pages="1-3")` to read first 3 pages
3. Extract key information systematically from ALL available content
4. Provide structured analysis following the BroadAxis-AI format below
5. **DO NOT include technical details like Priority, Confidence, Model Used, or Tokens in your response**

**IMPORTANT:** Since this is a PRIMARY document, you MUST read the entire document to capture all critical information.

## 📊 **Required Analysis Format:**

### **Document: {doc.filename}**

#### **What is This About?**

> A 3–5 sentence **plain-English overview** of the opportunity. Include:
> - Who issued it (organization)
> - What they need / are requesting  
> - Why (the business problem or goal)
> - Type of response expected (proposal, quote, info)

---

#### 🧩 **Key Opportunity Details**

**CRITICAL FORMATTING INSTRUCTIONS - YOU MUST FOLLOW EXACTLY:**

For EACH field below, you MUST format like this example:
```
- **Submission Deadline:**  
  October 24, 2025 at 2:00 PM CST
```

**RULES:**
1. Field name must be BOLD: `**Field Name:**`
2. Add TWO SPACES after the colon, then line break
3. Value goes on THE NEXT LINE (indented with 2 spaces)
4. Leave a BLANK line between each field

**Now list all of the following if available in the document:**

- **Submission Deadline:**  
  [Date + Time - be VERY specific: Month Day, Year at Time Timezone]

- **Project Start/End Dates:**  
  [Exact start/end dates, contract duration, renewal options]

- **Estimated Value / Budget:**  
  [If stated, include exact amount or budget range; if not stated, write "Not specified"]

- **Response Format:**  
  [PDF proposal, online portal, hard copy, email, etc.]

- **Delivery Location(s):**  
  [Full address, city, state, zip code, remote/on-site requirements]

- **Eligibility Requirements:**  
  [List each requirement on separate line with bullet points]

- **Scope Summary:**  
  [Comprehensive bullet points - use actual bullets for each service/deliverable]

- **Specific Technologies:**  
  [List each technology on its own line with bullets]

- **Insurance Requirements:**  
  [Each type on its own line with exact amounts - e.g., General Liability: $2M/$1M]

- **Staff Requirements:**  
  [Each requirement on its own line with bullets]

---

#### 📊 **Evaluation Criteria**

**How will responses be scored or selected?**

List evaluation criteria with weighting if provided. Format like this:
- Experience and Qualifications: 40%
- Technical Approach: 30%
- Price: 20%
- Local Preference: 10%

---

#### ⚠️ **Notable Risks or Challenges**

**List any red flags or items requiring clarification:**

Use bullet points for each risk/challenge:
- [Risk 1 - e.g., Tight timeline with only 30 days to respond]
- [Risk 2 - e.g., Vague scope without detailed requirements]
- [Risk 3 - e.g., Strict insurance requirements]

---

#### 💡 **Potential Opportunities or Differentiators**

**What gives us a competitive edge?**

Use bullet points for each opportunity:
- [Opportunity 1 - e.g., Optional services for additional revenue]
- [Opportunity 2 - e.g., Contract renewal options]
- [Opportunity 3 - e.g., Innovation clauses for differentiation]

---

#### 📞 **Contact & Submission Info**

- **Primary Contact:**  
  [Full name, title]  
  [Email address]  
  [Phone number]

- **Submission Instructions:**  
  [Detailed submission method - portal URL, email, physical address, etc.]

---

⚠️ **CRITICAL INSTRUCTIONS:**

1. **NO ERROR MESSAGES:** Do NOT include any messages like "Oops", "Let me try", "Hmm", "Ah, got it" in your output
2. **DIRECT ANALYSIS ONLY:** Start directly with the analysis content - skip any tool-calling commentary
3. **CONSISTENT FORMATTING:** Use the exact format above with proper line breaks after each field
4. **ONLY FACTS:** Only summarize what is clearly and explicitly stated - never guess or infer

**You must extract ALL specific details including:**
- Exact dates, times, and deadlines
- Specific technology names and versions  
- Insurance coverage amounts and types
- Background check and certification requirements
- Geographic and on-site requirements
- Contract terms and renewal options
- Pricing constraints and hold periods

Provide your analysis in the exact format above. Be thorough, specific, and comprehensive. Do not miss any important details.

**🔧 Tools Used:** extract_pdf_text"""
                    
                    # Process with token management and multi-model strategy
                    # Let token manager choose best model, but prefer Sonnet for comprehensive analysis
                    recommended_model = token_manager.get_recommended_model(analysis_prompt, 1000)
                    
                    # Override to Sonnet if token manager selected Haiku (Haiku's 4096 limit is too small)
                    if "haiku" in recommended_model.lower():
                        recommended_model = "claude-haiku-4-5-20251001"
                        logger.info("Overriding Haiku → Sonnet for RFP document analysis (needs >4096 output tokens)")
                    
                    result = await run_mcp_query(
                        analysis_prompt,
                        enabled_tools=['sharepoint_list_files', 'extract_pdf_text'],
                        model=recommended_model,
                        session_id=session_id
                    )
                    
                    analysis_result = {
                        "filename": doc.filename,
                        "priority": doc.priority.value,
                        "confidence": doc.confidence_score,
                        "analysis": result.get("response", ""),
                        "tokens_used": result.get("tokens_used", 0),
                        "model_used": result.get("model_used", "unknown"),
                        "cached": False
                    }
                    
                    processed_documents.append(analysis_result)
                    total_tokens_used += result.get("tokens_used", 0)
                    
                    # Store the analysis for future use
                    if SESSION_MANAGER_AVAILABLE:
                        await session_manager.store_document_summary(
                            session_id, 
                            doc.file_path, 
                            analysis_result
                        )
                
            except Exception as e:
                logger.error(f"Error processing primary document {doc.filename}: {e}")
                processed_documents.append({
                    "filename": doc.filename,
                    "priority": doc.priority.value,
                    "confidence": doc.confidence_score,
                    "analysis": f"Error processing document: {str(e)}",
                    "tokens_used": 0,
                    "model_used": "error",
                    "cached": False
                })
        
        # Step 5: Generate clean Go/No-Go analysis with caching
        # Create a cache key for the Go/No-Go analysis based on document content
        import hashlib
        analysis_cache_key = f"go_no_go_analysis:{hashlib.md5(''.join([doc['analysis'] for doc in processed_documents]).encode()).hexdigest()}"
        
        # Check if we have a cached Go/No-Go analysis
        summary_result = None
        if SESSION_MANAGER_AVAILABLE and session_manager.redis:
            cached_analysis = await session_manager.redis.get(analysis_cache_key)
            if cached_analysis:
                print(f"✅ Found cached Go/No-Go analysis")
                summary_result = {"response": json.loads(cached_analysis)}
        
        if not summary_result:
            # Generate new analysis
            go_no_go_prompt = f"""You are **BroadAxis-AI**, analyzing this RFP opportunity to provide a clear Go/No-Go recommendation.

**CRITICAL: You must be CONSISTENT. The same document analysis should ALWAYS produce the same Go/No-Go decision.**

## 📄 **Document Analysis Results**

{chr(10).join([f"### 📄 **{doc['filename']}**\n\n{remove_duplicate_doc_header(clean_markup(doc['analysis']), doc['filename'])}\n\n---\n" for doc in processed_documents])}

## 🧠 **Go/No-Go Analysis**

### 🔍 **Step 1: RFP Requirements Review**

> **CRITICAL:** Base your analysis ONLY on the document analysis results provided above. Do not hallucinate or make assumptions.

**Key Requirements Identified:**

- Highlight the most critical needs and evaluation criteria from the document analysis above
- Extract key deliverables, timeline, and scope requirements from the actual documents
- Identify any special compliance or certification requirements mentioned in the documents

---

### 🔎 **Step 2: Internal Knowledge Research**

Use `Broadaxis_knowledge_search` to research:
- Relevant past projects and similar work experience
- Proof of experience in the specific domain/industry
- Known strengths or capability gaps for this type of opportunity
- Government/public sector experience if applicable
- Geographic presence and local capabilities

**🎯 BroadAxis Strengths Identified:**

[Based on knowledge search results, list key capabilities and experience]

---

### ⚖️ **Step 3: Capability Alignment Assessment**

- **Capability Match:** BroadAxis meets ~[X]% of the requirements
- **Strong Areas:** [Specific competitive advantages]
- **Capability Gaps:** [Missing capabilities or unclear requirements]
- **Required Partnerships:** [Any subcontractor needs if applicable]

---

### 👥 **Step 4: Resource Requirements Analysis**

- **Specialized Skills Needed:** [List specific expertise required]
- **Timeline Feasibility:** [Can we meet proposal deadline and project timeline?]
- **Team Readiness:** [Current capacity and resource availability]
- **Partnership Needs:** [Required external support if any]

---

### 🏆 **Step 5: Competitive Positioning Evaluation**

- **Competitive Advantages:** [Local presence, certifications, experience, technology]
- **Competitive Challenges:** [Potential weaknesses vs competitors]
- **Win Probability:** [Realistic assessment based on known factors]

---

### 🚦 **FINAL RECOMMENDATION**

> **🎯 DECISION: [GO / NO-GO / CONDITIONAL-GO]**
> 
> **📝 RATIONALE:**  
> [2-3 sentences with clear, data-driven reasoning based on the analysis above]
> 
> **🎯 CONFIDENCE LEVEL:** [High / Medium / Low]
> 
> **📊 SUCCESS PROBABILITY:** [X]% with proper preparation

---

## 📋 **Action Plan** *(if GO/CONDITIONAL-GO)*

### **⚡ Immediate Actions (Next 7 Days):**

1. [Specific capability assessment or information gathering task]
2. [Key stakeholder meetings or decisions required]
3. [Critical preparation activities]

---

### **📝 RFP Response Preparation (Week 2):**

1. [Proposal development and writing tasks]
2. [Technical solution design and validation]
3. [Cost estimation and pricing strategy]

---

### **⚠️ Risk Mitigation Strategies:**

1. [How to address capability gaps]
2. [How to manage timeline constraints]
3. [How to strengthen competitive position]

**Success Probability:** [X]% with proper preparation, versus [Y]% without focused effort.

---

## 📄 **Documents to Create**

Based on the RFP requirements identified above, list ALL documents that need to be created for submission.

---

### **Step 1 - Extract Required Documents:**

Review the document analysis above and identify every submission document mentioned in the RFP, including:
- Technical proposals, executive summaries, company profiles
- Past performance references, project examples, case studies
- Staff resumes, organizational charts, team structures
- Pricing sheets, cost breakdowns, budget justifications
- Compliance certifications, insurance certificates, licenses
- Project plans, timelines, work breakdown structures, methodologies
- Quality assurance plans, security plans, risk management plans
- Any other required attachments, forms, or exhibits

---

### **Step 2 - Assess BroadAxis Information Availability:**

For EACH document identified above, you MUST use `Broadaxis_knowledge_search` to determine if BroadAxis has the necessary information/data to create it:
- Search for relevant company capabilities, past projects, certifications
- Check for existing templates, previous proposals, company documentation
- Verify availability of required data (financials, staff info, technical specs)

---

### **Step 3 - Classify Each Document:**

- **✅ Complete Information:** BroadAxis has all necessary data to create this document
- **🟡 Partial Information:** BroadAxis has some data but missing key details
- **❌ No Information:** BroadAxis lacks the necessary data to create this document

---

### **📋 Output Format:**

| # | Document Name | Information Status | What We Have / What's Missing |
|---|--------------|-------------------|-------------------------------|
| 1 | [Exact document name from RFP] | ✅ Complete / 🟡 Partial / ❌ No Info | [Brief explanation of available data or gaps] |
| 2 | [Document name] | [Status] | [Details] |

---

**CRITICAL REQUIREMENTS:**
- ✅ List ALL required submission documents - do not miss any
- ✅ Base document names on actual RFP requirements, not assumptions
- ✅ Use `Broadaxis_knowledge_search` to verify what information exists in the knowledge base
- ✅ Be specific about what's available and what's missing
- ✅ If a document type is mentioned but details are unclear, still list it and mark as "Partial"

---

## ⚠️ **Important Guidelines:**

1. **NO ERROR MESSAGES:** Do NOT include any "Oops", "Let me try", "Hmm" messages
2. **NO CONTINUATION MESSAGES:** Do NOT include "Continued in next part" or "[Continued...]" - provide COMPLETE analysis
3. **VERIFIED INFO ONLY:** Use only verified internal information via `Broadaxis_knowledge_search`
4. **NO HALLUCINATIONS:** Only reference requirements, technologies, and details explicitly mentioned in the document analysis
5. **BE SPECIFIC:** Clearly state what information exists and what's missing
6. **ACTIONABLE:** If recommendation is GO, list specific tasks for RFP submission
7. **COMPLETE OUTPUT:** Provide the FULL analysis in one response - do not truncate or continue later

**ANTI-HALLUCINATION RULE:** Do not mention BI, data warehousing, or other technologies unless they appear in the actual RFP documents.

---

**🔧 Tools to Use:** Broadaxis_knowledge_search

**CRITICAL:** You have 8,192 output tokens available. This is sufficient for a COMPLETE analysis. Do not truncate or add continuation messages like "[Continued...]". Provide the full, comprehensive analysis in this single response.

Provide your analysis in the exact format above with proper line breaks and clear section separators. Be thorough, data-driven, and actionable."""
        
            # Use Sonnet for Go/No-Go analysis (needs more output tokens for complete document list)
            recommended_model = token_manager.get_recommended_model(go_no_go_prompt, 2000)
            
            # Override to Sonnet if token manager selected Haiku (need complete document list with action plan)
            if "haiku" in recommended_model.lower():
                recommended_model = "claude-haiku-4-5-20251001"
                logger.info("Overriding Haiku → Sonnet for Go/No-Go analysis (needs >4096 output tokens)")
            
            summary_result = await run_mcp_query(
                go_no_go_prompt,
                enabled_tools=['Broadaxis_knowledge_search'],
                model=recommended_model,
                session_id=session_id
            )
            
            # Cache the Go/No-Go analysis for consistency
            if SESSION_MANAGER_AVAILABLE and session_manager.redis and summary_result.get("response"):
                await session_manager.redis.setex(analysis_cache_key, 604800, json.dumps(summary_result["response"]))  # 7 days
                print(f"✅ Cached Go/No-Go analysis for consistency")
        
        total_tokens_used += summary_result.get("tokens_used", 0)
        
        # Store the complete RFP analysis for future reference
        rfp_analysis = {
            "folder_path": folder_path,
            "total_documents": len(documents),
            "processed_documents": len(processed_documents),
            "recommendation": recommendation,
            "processed_docs": processed_documents,
            "summary": summary_result.get("response", ""),
            "total_tokens_used": total_tokens_used,
            "timestamp": datetime.now().isoformat()
        }
        
        if SESSION_MANAGER_AVAILABLE:
            await session_manager.store_rfp_analysis(session_id, rfp_analysis)
        
        # Prepare classification lists from prioritization (not only processed docs)
        primary_names = [d.filename for d in primary_docs]
        secondary_names = [d.filename for d in secondary_docs]
        prioritized_filenames = {d.filename for d in prioritized_docs}
        non_primary_secondary = set(primary_names) | set(secondary_names)
        other_names = [name for name in prioritized_filenames if name not in non_primary_secondary]

        # Counts per category (reflect all files discovered in folder)
        primary_count = len(primary_names)
        secondary_count = len(secondary_names)
        other_count = len(other_names)

        # Clean simple HTML-like tags the LLM may emit (e.g., <result>) and any anchors
        

        # Create clean, concise response without redundancy
        # Group documents by category for display
        primary_docs = [doc for doc in processed_documents if doc.get('priority') == 'primary']
        secondary_docs = [doc for doc in processed_documents if doc.get('priority') == 'secondary']
        other_docs = [doc for doc in processed_documents if doc.get('priority') == 'other']
        
        def format_doc_names(docs):
            if not docs:
                return "❌ None"
            return " • ".join([doc['filename'] for doc in docs])
        
        formatted_response = f"""**Intelligent RFP Processing Complete**

## 📊 **Document Analysis**

| **Category** | **Count** | **Documents** |
|--------------|-----------|---------------|
| 🎯 **Primary Documents** | `{primary_count}` files | {format_doc_names(primary_docs)} |
| 📋 **Secondary Documents** | `{secondary_count}` files | {format_doc_names(secondary_docs)} |
| 📄 **Other Documents** | `{other_count}` files | {format_doc_names(other_docs)} |
| **📊 Total Processed** | `{len(processed_documents)}` files | ✅ Complete |

---

## 📄 **Document Summaries**

{chr(10).join([f"### 📄 **{doc['filename']}**\n\n{remove_duplicate_doc_header(clean_markup(doc['analysis']), doc['filename'])}\n" for doc in processed_documents])}

---

## 🎯 **FINAL RECOMMENDATION**

{clean_markup(summary_result.get("response", ""))}

---

> **💡 This recommendation is based on comprehensive analysis of the RFP requirements and BroadAxis capabilities.**
"""

        # Store the RFP processing conversation in the session
        if SESSION_MANAGER_AVAILABLE:
            # Store user message
            user_message = {
                "role": "user",
                "content": f"Process RFP folder intelligently: {folder_path}",
                "timestamp": datetime.now().isoformat()
            }
            await session_manager.add_message(session_id, user_message)
            
            # Store AI response
            ai_message = {
                "role": "assistant",
                "content": formatted_response,
                "timestamp": datetime.now().isoformat()
            }
            await session_manager.add_message(session_id, ai_message)
            print(f"✅ Stored RFP processing conversation in session {session_id}")

        return {
            "status": "success",
            "folder_path": folder_path,
            "total_documents": len(documents),
            "processed_documents": len(processed_documents),
            "recommendation": recommendation,
            "processed_docs": processed_documents,
            "summary": formatted_response,
            "total_tokens_used": total_tokens_used,
            "token_breakdown": {
                "total_tokens": total_tokens_used,
                "input_tokens": int(total_tokens_used * 0.7),  # Estimate
                "output_tokens": int(total_tokens_used * 0.3),  # Estimate
                "model_used": "claude-haiku-4-5-20251001",  # Default for RFP processing
                "queries_processed": len(processed_documents) + 1  # Documents + Go/No-Go analysis
            },
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_handler.log_error(e, {'operation': 'process_rfp_folder_intelligent', 'folder_path': folder_path})
        return {
            "status": "error",
            "message": f"Error processing RFP folder: {str(e)}",
            "folder_path": folder_path
        }


# Authentication Endpoints
@app.post("/api/auth/register", response_model=AuthResponse)
async def register_user(user_data: UserRegister):
    """Register a new user"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Check if user already exists
        existing_user = await session_manager.redis.get(f"user:email:{user_data.email}")
        if existing_user:
            return JSONResponse(
                status_code=400,
                content={"error": "User with this email already exists"}
            )
        
        # Create new user
        user_id = str(uuid.uuid4())
        hashed_password = hash_password(user_data.password)
        now = datetime.now().isoformat()
        
        user = {
            "id": user_id,
            "name": user_data.name,
            "email": user_data.email,
            "password_hash": hashed_password,
            "created_at": now,
            "last_login": None
        }
        
        # Store user in Redis
        await session_manager.redis.setex(f"user:{user_id}", 86400 * 30, json.dumps(user))  # 30 days
        await session_manager.redis.setex(f"user:email:{user_data.email}", 86400 * 30, user_id)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_id}, expires_delta=access_token_expires
        )
        
        # Store session in Redis
        session_data = {
            "user_id": user_id,
            "created_at": now,
            "expires_at": (datetime.now() + access_token_expires).isoformat()
        }
        await session_manager.redis.setex(f"session:{access_token}", int(access_token_expires.total_seconds()), json.dumps(session_data))
        
        print(f"✅ User registered: {user_data.email} (ID: {user_id})")
        
        return AuthResponse(
            access_token=access_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(
                id=user_id,
                name=user_data.name,
                email=user_data.email,
                created_at=now
            )
        )
        
    except Exception as e:
        error_handler.log_error(e, {'operation': 'register_user', 'email': user_data.email})
        return JSONResponse(
            status_code=500,
            content={"error": "Registration failed"}
        )

@app.post("/api/auth/login", response_model=AuthResponse)
async def login_user(user_data: UserLogin):
    """Login a user"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Get user by email
        user_id = await session_manager.redis.get(f"user:email:{user_data.email}")
        if not user_id:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid email or password"}
            )
        
        # Get user data
        user_data_str = await session_manager.redis.get(f"user:{user_id}")
        if not user_data_str:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid email or password"}
            )
        
        user = json.loads(user_data_str)
        
        # Verify password
        if not verify_password(user_data.password, user["password_hash"]):
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid email or password"}
            )
        
        # Update last login
        now = datetime.now().isoformat()
        user["last_login"] = now
        await session_manager.redis.setex(f"user:{user_id}", 86400 * 30, json.dumps(user))
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_id}, expires_delta=access_token_expires
        )
        
        # Store session in Redis
        session_data = {
            "user_id": user_id,
            "created_at": now,
            "expires_at": (datetime.now() + access_token_expires).isoformat()
        }
        await session_manager.redis.setex(f"session:{access_token}", int(access_token_expires.total_seconds()), json.dumps(session_data))
        
        print(f"✅ User logged in: {user_data.email} (ID: {user_id})")
        
        return AuthResponse(
            access_token=access_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(
                id=user["id"],
                name=user["name"],
                email=user["email"],
                created_at=user["created_at"],
                last_login=now
            )
        )

    except Exception as e:
        error_handler.log_error(e, {'operation': 'login_user', 'email': user_data.email})
        return JSONResponse(
            status_code=500,
            content={"error": "Login failed"}
        )

@app.post("/api/auth/forgot")
async def forgot_password(req: ForgotPasswordRequest):
    """Initiate password reset: create a one-time token with short TTL and (in dev) return it."""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(status_code=503, content={"error": "Session management not available"})
        if not session_manager.redis:
            await session_manager.connect()

        # Lookup user id by email
        user_id = await session_manager.redis.get(f"user:email:{req.email}")
        # Always respond success to avoid user enumeration
        if not user_id:
            return {"status": "ok"}

        # Create reset token
        reset_token = str(uuid.uuid4())
        # Store mapping token -> user_id with 15-minute TTL
        await session_manager.redis.setex(f"password_reset:{reset_token}", 900, user_id)

        # In production: send email with reset link including token.
        # For development: return token in response so user can reset immediately.
        return {"status": "ok", "reset_token": reset_token}
    except Exception as e:
        error_handler.log_error(e, {'operation': 'forgot_password'})
        return JSONResponse(status_code=500, content={"error": "Failed to initiate password reset"})

@app.post("/api/auth/reset")
async def reset_password(req: ResetPasswordRequest):
    """Complete password reset given a valid token."""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(status_code=503, content={"error": "Session management not available"})
        if not session_manager.redis:
            await session_manager.connect()

        # Resolve token -> user_id
        user_id = await session_manager.redis.get(f"password_reset:{req.token}")
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "Invalid or expired token"})

        # Get user data
        user_data_str = await session_manager.redis.get(f"user:{user_id}")
        if not user_data_str:
            return JSONResponse(status_code=404, content={"error": "User not found"})
        user = json.loads(user_data_str)

        # Update password hash
        new_hash = hash_password(req.new_password)
        user["password_hash"] = new_hash
        await session_manager.redis.setex(f"user:{user_id}", 86400 * 30, json.dumps(user))

        # Invalidate token
        await session_manager.redis.delete(f"password_reset:{req.token}")

        return {"status": "success"}
    except Exception as e:
        error_handler.log_error(e, {'operation': 'reset_password'})
        return JSONResponse(status_code=500, content={"error": "Failed to reset password"})

@app.post("/api/auth/logout")
async def logout_user(request: Request):
    """Logout a user"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Get token from Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "No valid token provided"}
            )
        
        token = authorization.split(" ")[1]
        
        # Remove session from Redis
        await session_manager.redis.delete(f"session:{token}")
        
        print(f"✅ User logged out")
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        error_handler.log_error(e, {'operation': 'logout_user'})
        return JSONResponse(
            status_code=500,
            content={"error": "Logout failed"}
        )

@app.get("/api/auth/users", response_model=List[UserResponse])
async def get_all_users(current_user: UserResponse = Depends(get_current_user)):
    """Get all registered users"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Get all user keys - try different patterns
        user_keys = await session_manager.redis.keys("user:*")
        print(f"🔍 Found {len(user_keys)} user keys: {user_keys}")
        
        # Also check for any other user-related keys
        all_keys = await session_manager.redis.keys("*")
        user_related_keys = [key for key in all_keys if "user" in key.lower()]
        print(f"🔍 All user-related keys: {user_related_keys}")
        
        # Filter out email keys (user:email:*) but keep user:uuid keys
        user_id_keys = [key for key in user_keys if not key.startswith("user:email:")]
        print(f"🔍 Filtered to {len(user_id_keys)} user ID keys: {user_id_keys}")
        
        # If we don't find enough users, let's also check for session keys that might have user info
        if len(user_id_keys) <= 1:
            session_keys = await session_manager.redis.keys("session:*")
            print(f"🔍 Found {len(session_keys)} session keys")
            for session_key in session_keys[:5]:  # Check first 5 sessions
                session_data = await session_manager.redis.get(session_key)
                if session_data:
                    session = json.loads(session_data)
                    if 'user_id' in session:
                        print(f"🔍 Session {session_key} has user_id: {session['user_id']}")
                        # Try to get user data for this user_id
                        user_data = await session_manager.redis.get(f"user:{session['user_id']}")
                        if user_data:
                            user = json.loads(user_data)
                            print(f"🔍 Found user from session: {user.get('name', 'Unknown')}")
        
        users = []
        seen_user_ids = set()
        
        # First, process the direct user keys
        for key in user_id_keys:
            user_data_str = await session_manager.redis.get(key)
            if user_data_str:
                user = json.loads(user_data_str)
                user_id = user.get("id")
                if user_id and user_id not in seen_user_ids:
                    seen_user_ids.add(user_id)
                    print(f"🔍 Found user: {user.get('name', 'Unknown')} ({user.get('email', 'No email')})")
                    # Don't include password hash in response
                    user_response = UserResponse(
                        id=user["id"],
                        name=user["name"],
                        email=user["email"],
                        created_at=user["created_at"],
                        last_login=user.get("last_login")
                    )
                    users.append(user_response)
            else:
                print(f"⚠️ No data found for key: {key}")
        
        # Also check all email keys to find additional users
        email_keys = [key for key in user_keys if key.startswith("user:email:")]
        print(f"🔍 Checking {len(email_keys)} email keys for additional users")
        
        for email_key in email_keys:
            user_id = await session_manager.redis.get(email_key)
            if user_id and user_id not in seen_user_ids:
                user_data_str = await session_manager.redis.get(f"user:{user_id}")
                if user_data_str:
                    user = json.loads(user_data_str)
                    seen_user_ids.add(user_id)
                    print(f"🔍 Found user via email key: {user.get('name', 'Unknown')} ({user.get('email', 'No email')})")
                    user_response = UserResponse(
                        id=user["id"],
                        name=user["name"],
                        email=user["email"],
                        created_at=user["created_at"],
                        last_login=user.get("last_login")
                    )
                    users.append(user_response)
        
        print(f"🔍 Returning {len(users)} users")
        
        return users
        
    except Exception as e:
        print(f"❌ Error getting users: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get users"}
        )

# Task Assignment Models
class TaskAssignment(BaseModel):
    id: str
    category: str  # Project, Meeting, Internal, Review, Other, Document Creation
    type: str  # RFP, RFI, RFQ, Meeting, Document Review, etc.
    title: str  # Task title/description
    document: str = ""  # SharePoint path (optional)
    assigned_to: str  # User name
    assigned_by: str  # User who assigned it
    status: str  # Assigned, Review, In Progress, Completed
    priority: str = "Medium"  # High, Medium, Low
    due_date: str = ""  # Optional due date
    decision: str = "Decision Pending"  # Go, No-Go, Decision Pending (for RFP/RFI/RFQ)
    created_at: str
    updated_at: str
    parent_task_id: Optional[str] = None  # For document tasks linked to parent RFP
    parent_rfp_path: Optional[str] = None  # e.g., "RFP/Dallas City"
    document_details: Optional[str] = None  # Details about the document requirement
    original_status: Optional[str] = None  # ✅/🟡/❌ status from AI
    document_count: Optional[int] = None  # For parent tasks - how many documents needed

class TaskAssignmentRequest(BaseModel):
    category: str
    type: str
    title: str
    document: str = ""
    assigned_to: str
    status: str = "Assigned"
    priority: str = "Medium"
    due_date: str = ""
    decision: str = "Decision Pending"

# Task Assignment Endpoints
@app.get("/api/tasks", response_model=List[TaskAssignment])
async def get_all_tasks(current_user: UserResponse = Depends(get_current_user)):
    """Get all task assignments"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Get all task keys
        task_keys = await session_manager.redis.keys("task:*")
        print(f"🔍 Found {len(task_keys)} task keys")
        
        tasks = []
        for key in task_keys:
            task_data_str = await session_manager.redis.get(key)
            if task_data_str:
                task = json.loads(task_data_str)
                tasks.append(TaskAssignment(**task))
        
        # Sort by created_at (newest first)
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        print(f"🔍 Returning {len(tasks)} tasks")
        return tasks
        
    except Exception as e:
        print(f"❌ Error getting tasks: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get tasks"}
        )

@app.post("/api/tasks", response_model=TaskAssignment)
async def create_task_assignment(
    task_data: TaskAssignmentRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Create a new task assignment"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Create task ID
        task_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Create task assignment
        task = TaskAssignment(
            id=task_id,
            category=task_data.category,
            type=task_data.type,
            title=task_data.title,
            document=task_data.document,
            assigned_to=task_data.assigned_to,
            assigned_by=current_user.name,
            status=task_data.status,
            priority=task_data.priority,
            due_date=task_data.due_date,
            decision=task_data.decision,
            created_at=now,
            updated_at=now
        )
        
        # Store in Redis
        await session_manager.redis.setex(
            f"task:{task_id}", 
            86400 * 365,  # 1 year TTL
            json.dumps(task.dict())
        )
        
        print(f"✅ Created task assignment: {task_id} for {task_data.assigned_to}")
        return task
        
    except Exception as e:
        print(f"❌ Error creating task: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to create task assignment"}
        )

@app.put("/api/tasks/{task_id}/status")
async def update_task_status(
    task_id: str,
    status: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Update task status"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Get existing task
        task_data_str = await session_manager.redis.get(f"task:{task_id}")
        if not task_data_str:
            return JSONResponse(
                status_code=404,
                content={"error": "Task not found"}
            )
        
        task = json.loads(task_data_str)
        task["status"] = status
        task["updated_at"] = datetime.now().isoformat()
        
        # Update in Redis
        await session_manager.redis.setex(
            f"task:{task_id}", 
            86400 * 365,  # 1 year TTL
            json.dumps(task)
        )
        
        print(f"✅ Updated task {task_id} status to {status}")
        return {"status": "success", "message": "Task status updated"}
        
    except Exception as e:
        print(f"❌ Error updating task status: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to update task status"}
        )

@app.put("/api/tasks/{task_id}/decision")
async def update_task_decision(
    task_id: str,
    decision: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Update task decision (Go/No-Go/Decision Pending)"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Get existing task
        task_data_str = await session_manager.redis.get(f"task:{task_id}")
        if not task_data_str:
            return JSONResponse(
                status_code=404,
                content={"error": "Task not found"}
            )
        
        task = json.loads(task_data_str)
        task["decision"] = decision
        task["updated_at"] = datetime.now().isoformat()
        
        # Update in Redis
        await session_manager.redis.setex(
            f"task:{task_id}", 
            86400 * 365,  # 1 year TTL
            json.dumps(task)
        )
        
        print(f"✅ Updated task {task_id} decision to {decision}")
        return {"status": "success", "message": "Task decision updated"}
        
    except Exception as e:
        print(f"❌ Error updating task decision: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to update task decision"}
        )

@app.delete("/api/tasks/cleanup")
async def cleanup_old_completed_tasks(
    current_user: UserResponse = Depends(get_current_user)
):
    """Delete completed tasks older than 7 days"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        from datetime import datetime, timedelta
        
        # Calculate cutoff date (7 days ago)
        cutoff_date = datetime.now() - timedelta(days=7)
        cutoff_iso = cutoff_date.isoformat()
        
        # Get all task keys
        task_keys = await session_manager.redis.keys("task:*")
        deleted_count = 0
        
        for task_key in task_keys:
            task_data = await session_manager.redis.get(task_key)
            if task_data:
                task = json.loads(task_data)
                
                # Check if task is completed and older than 7 days
                if (task.get('status') == 'Completed' and 
                    task.get('updated_at', '') < cutoff_iso):
                    
                    # Delete the task
                    await session_manager.redis.delete(task_key)
                    deleted_count += 1
                    print(f"🗑️ Deleted old completed task: {task.get('title', 'Unknown')}")
        
        print(f"✅ Cleanup completed. Deleted {deleted_count} old completed tasks.")
        return {
            "status": "success",
            "message": f"Cleanup completed. Deleted {deleted_count} old completed tasks.",
            "deleted_count": deleted_count,
            "cutoff_date": cutoff_iso
        }
        
    except Exception as e:
        print(f"❌ Error cleaning up old tasks: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to cleanup old tasks"}
        )

@app.delete("/api/tasks/{task_id}")
async def delete_task(
    task_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Delete a specific task"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Check if task exists
        task_key = f"task:{task_id}"
        task_data = await session_manager.redis.get(task_key)
        
        if not task_data:
            return JSONResponse(
                status_code=404,
                content={"error": "Task not found"}
            )
        
        task = json.loads(task_data)
        
        # Optional: Check if user has permission to delete (e.g., assigned by them or admin)
        # For now, allow any authenticated user to delete any task
        # You can add permission checks here if needed
        
        # Delete the task
        await session_manager.redis.delete(task_key)
        
        print(f"🗑️ Deleted task: {task.get('title', 'Unknown')} (ID: {task_id})")
        return {
            "status": "success",
            "message": "Task deleted successfully",
            "deleted_task": task.get('title', 'Unknown')
        }
        
    except Exception as e:
        print(f"❌ Error deleting task: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to delete task"}
        )

# Import RFP Documents as Tasks
class RFPDocumentImportRequest(BaseModel):
    parent_task_id: str  # ID of the parent RFP task
    parent_rfp_path: str  # e.g., "RFP/Dallas City"
    documents: List[dict]  # List of documents with name, status, details

@app.post("/api/tasks/import-rfp-documents")
async def import_rfp_documents(
    request: RFPDocumentImportRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Import RFP required documents as sub-tasks"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Verify parent task exists
        parent_task_data = await session_manager.redis.get(f"task:{request.parent_task_id}")
        if not parent_task_data:
            return JSONResponse(
                status_code=404,
                content={"error": "Parent task not found"}
            )
        
        parent_task = json.loads(parent_task_data)
        now = datetime.now().isoformat()
        
        created_tasks = []
        
        for doc in request.documents:
            # Map status emoji to task status and priority
            doc_status = doc.get('status', '')
            if '✅' in doc_status or 'Complete' in doc_status:
                task_status = "Assigned"
                task_priority = "Low"  # We have the info
            elif '🟡' in doc_status or 'Partial' in doc_status:
                task_status = "Assigned"
                task_priority = "Medium"  # Need to gather some info
            else:  # ❌ or No Info
                task_status = "Assigned"
                task_priority = "High"  # Need to gather all info
            
            # Create task ID
            task_id = str(uuid.uuid4())
            
            # Create document task
            task = TaskAssignment(
                id=task_id,
                category="Document Creation",
                type=doc.get('name', 'Unknown Document'),
                title=f"{doc.get('name', 'Unknown Document')} - {request.parent_rfp_path}",
                document=request.parent_rfp_path,
                assigned_to=parent_task.get('assigned_to', current_user.name),
                assigned_by=current_user.name,
                status=task_status,
                priority=task_priority,
                due_date=parent_task.get('due_date', ''),
                decision='N/A',  # Document tasks don't need Go/No-Go
                created_at=now,
                updated_at=now
            )
            
            # Store in Redis with parent task reference
            task_dict = task.dict()
            task_dict['parent_task_id'] = request.parent_task_id
            task_dict['parent_rfp_path'] = request.parent_rfp_path
            task_dict['document_details'] = doc.get('details', '')
            task_dict['original_status'] = doc.get('status', '')
            
            await session_manager.redis.setex(
                f"task:{task_id}",
                86400 * 365,  # 1 year TTL
                json.dumps(task_dict)
            )
            
            created_tasks.append(task)
            print(f"✅ Created document task: {doc.get('name')} for {request.parent_rfp_path}")
        
        # Update parent task with document count
        parent_task['document_count'] = len(created_tasks)
        parent_task['updated_at'] = now
        await session_manager.redis.setex(
            f"task:{request.parent_task_id}",
            86400 * 365,
            json.dumps(parent_task)
        )
        
        return {
            "status": "success",
            "message": f"Imported {len(created_tasks)} document tasks",
            "created_tasks": created_tasks,
            "parent_task_id": request.parent_task_id
        }
        
    except Exception as e:
        print(f"❌ Error importing RFP documents: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to import RFP documents"}
        )

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user(request: Request):
    """Get current user information"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"error": "Session management not available"}
            )
        
        # Ensure Redis connection
        if not session_manager.redis:
            await session_manager.connect()
        
        # Get token from Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "No valid token provided"}
            )
        
        token = authorization.split(" ")[1]
        
        # Verify token and get user_id
        user_id = verify_token(token)
        if not user_id:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid token"}
            )
        
        # Check if session exists in Redis
        session_data = await session_manager.redis.get(f"session:{token}")
        if not session_data:
            return JSONResponse(
                status_code=401,
                content={"error": "Session expired"}
            )
        
        # Get user data
        user_data_str = await session_manager.redis.get(f"user:{user_id}")
        if not user_data_str:
            return JSONResponse(
                status_code=404,
                content={"error": "User not found"}
            )
        
        user = json.loads(user_data_str)
        
        return UserResponse(
            id=user["id"],
            name=user["name"],
            email=user["email"],
            created_at=user["created_at"],
            last_login=user.get("last_login")
        )
        
    except Exception as e:
        error_handler.log_error(e, {'operation': 'get_current_user'})
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get user information"}
        )

from fastapi import Form  # ← add this import

@app.post("/api/upload-local")
async def upload_local(file: UploadFile = File(...), session_id: str = Form("default")):
    """
    Accept a local file, extract text, chunk, and store by session_id.
    Returns: {doc_id, filename, pages, text_preview}
    """
    filename = file.filename or "upload"
    ext = os.path.splitext(filename.lower())[1]
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Uploaded file is empty")
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(400, "File exceeds 50MB limit")

    # Extract into pages list
    if ext == ".pdf":
        pages = _pdf_to_pages(raw)
        full_text = "\n".join(pages)
    elif ext in {".docx", ".doc"}:
        full_text = _docx_to_text(raw)
        pages = [full_text]  # docx/doc: treat as one big page for now
    else:  # .txt/.md
        full_text = _safe_text(raw.decode("utf-8", errors="ignore"))
        pages = full_text.split("\f") if "\f" in full_text else [full_text]

    chunks = _split_into_chunks(pages)
    doc_id = str(uuid.uuid4())

    # store in-process
    if session_id not in UPLOAD_STORE:
        UPLOAD_STORE[session_id] = {}
    UPLOAD_STORE[session_id][doc_id] = {
        "filename": filename,
        "pages": len(pages),
        "chunks": chunks,
        "preview": full_text[:PREVIEW_CHARS]
    }

    return {
        "doc_id": doc_id,
        "filename": filename,
        "pages": len(pages),
        "text_preview": full_text[:PREVIEW_CHARS]
    }

@app.get("/api/upload-local/{doc_id}/text")
async def get_uploaded_text(doc_id: str, session_id: str):
    """
    Optional helper: return concatenated text (capped) for debugging/continuations.
    """
    sess = UPLOAD_STORE.get(session_id, {})
    doc = sess.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found in this session")
    joined = "\n\n".join(c["text"] for c in doc["chunks"])
    return {"doc_id": doc_id, "session_id": session_id, "text": joined[:200000]}

@app.post("/api/upload-local/{doc_id}/search")
async def search_uploaded_doc(doc_id: str, payload: LocalSearchRequest):
    """
    Lightweight retrieval over stored chunks.
    Body: {session_id, query, k}
    Returns: {chunks: [{page_start, page_end, text, score}]}
    """
    sess = UPLOAD_STORE.get(payload.session_id, {})
    doc = sess.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found in this session")

    chunks = doc["chunks"]
    k = max(1, min(int(payload.k or 8), MAX_RETURN_CHUNKS))
    scored = _score_chunks(payload.query, chunks)
    top = [(s, c) for (s, c) in scored[:k]]

    # If no overlap at all, still return first k chunks so UI has context
    if not any(s > 0 for s, _ in top) and chunks:
        top = [(0.0, c) for c in chunks[:k]]

    return {
        "chunks": [
            {
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "text": c["text"][:4000],  # trim per-chunk for payload size
                "score": float(s)
            }
            for (s, c) in top
        ]
    }


@app.post("/api/upload-folder")
async def upload_folder(
    files: List[UploadFile] = File(...),
    target_folder: str = Form(...),
    session_id: str = Form("default"),
    current_user: UserResponse = Depends(get_current_user)
):
    """Upload multiple files to SharePoint folder maintaining directory structure"""
    try:
        if not files:
            return JSONResponse(
                status_code=400,
                content={"error": "No files provided"}
            )
        
        if target_folder not in ['RFP', 'RFI', 'RFQ']:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid target folder. Must be RFP, RFI, or RFQ"}
            )
        
        # Handle null session_id from frontend
        if session_id == "null" or not session_id:
            session_id = "default"
        
        upload_results = []
        sharepoint_manager = SharePointManager()
        
        for file in files:
            if not file.filename:
                print(f"⚠️ Skipping file with no filename")
                continue
                
            print(f"📁 Processing file: {file.filename}")
            
            # Validate file type
            allowed_extensions = {'.pdf', '.txt', '.md', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt'}
            file_ext = os.path.splitext(file.filename.lower())[1]
            print(f"🔍 File extension: {file_ext}")
            
            if file_ext not in allowed_extensions:
                error_msg = f"File type {file_ext} not supported"
                print(f"❌ {error_msg}")
                upload_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": error_msg
                })
                continue
            
            try:
                file_content = await file.read()
                print(f"📊 File size: {len(file_content)} bytes")
                
                if not file_content:
                    error_msg = "File is empty"
                    print(f"❌ {error_msg}")
                    upload_results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": error_msg
                    })
                    continue
                
                # Handle folder structure - when using webkitdirectory, filename contains the full path
                # Extract just the filename and create the proper folder path
                if '/' in file.filename:
                    # Split the path and filename
                    path_parts = file.filename.split('/')
                    filename_only = path_parts[-1]  # Get just the filename
                    folder_structure = '/'.join(path_parts[:-1])  # Get the folder path
                    
                    # Create the full target path: target_folder/folder_structure
                    full_target_path = f"{target_folder}/{folder_structure}" if folder_structure else target_folder
                    print(f"📂 Folder structure detected: {folder_structure}")
                    print(f"📄 Filename only: {filename_only}")
                    print(f"🎯 Target path: {full_target_path}")
                else:
                    # No folder structure, just use the filename and target folder
                    filename_only = file.filename
                    full_target_path = target_folder
                    print(f"📄 No folder structure, using filename: {filename_only}")
                    print(f"🎯 Target path: {full_target_path}")
                
                print(f"🚀 Uploading to SharePoint...")
                # Upload to SharePoint with proper folder structure
                upload_result = sharepoint_manager.upload_file_to_sharepoint(
                    file_content, 
                    filename_only, 
                    full_target_path
                )
                print(f"📋 Upload result: {upload_result}")
                
                if upload_result.get('status') == 'success':
                    upload_results.append({
                        "filename": file.filename,
                        "uploaded_filename": filename_only,
                        "target_path": full_target_path,
                        "status": "success",
                        "sharepoint_url": upload_result.get('web_url'),
                        "size": len(file_content)
                    })
                else:
                    upload_results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": upload_result.get('error', 'Upload failed')
                    })
                    
            except Exception as e:
                upload_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        successful_uploads = [r for r in upload_results if r["status"] == "success"]
        failed_uploads = [r for r in upload_results if r["status"] == "error"]
        
        print(f"📊 Upload Summary:")
        print(f"   Total files: {len(files)}")
        print(f"   Successful: {len(successful_uploads)}")
        print(f"   Failed: {len(failed_uploads)}")
        print(f"   Failed details: {failed_uploads}")
        
        return {
            "status": "success" if successful_uploads else "partial_success" if upload_results else "error",
            "target_folder": target_folder,
            "total_files": len(files),
            "successful_uploads": len(successful_uploads),
            "failed_uploads": len(failed_uploads),
            "results": upload_results,
            "message": f"Uploaded {len(successful_uploads)}/{len(files)} files to {target_folder} directory with folder structure preserved"
        }
        
    except Exception as e:
        error_handler.log_error(e, {
            'operation': 'upload_folder',
            'target_folder': target_folder,
            'file_count': len(files) if files else 0
        })
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to upload folder",
                "details": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

