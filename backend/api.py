

import asyncio
import json
import os
import time
import base64
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

import nest_asyncio
from fastapi import FastAPI, File, UploadFile, WebSocket, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ValidationError as PydanticValidationError
from dotenv import load_dotenv


# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

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
from sharepoint_api import sharepoint_router

# Import session manager (optional for now)
try:
    from session_manager import session_manager
    SESSION_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Session manager not available: {e}")
    SESSION_MANAGER_AVAILABLE = False
    session_manager = None



nest_asyncio.apply()

app = FastAPI(title="BroadAxis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(email_router)
app.include_router(sharepoint_router)

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

class RFPProcessingRequest(BaseModel):
    folder_path: str
    session_id: str = "default"

# Session file storage (keeping for backward compatibility)
session_files = {}

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
async def chat_with_context(request: ChatRequest, session_id: str = None):
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


@app.post("/api/session/create")
async def create_session():
    """Create a new session"""
    try:
        if not SESSION_MANAGER_AVAILABLE:
            return {
                "session_id": None,
                "error": "Session management not available",
                "status": "error"
            }
        
        session_id = await session_manager.create_session()
        return {
            "session_id": session_id,
            "status": "success"
        }
    except Exception as e:
        error_handler.log_error(e, {'operation': 'create_session'})
        raise

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
async def process_rfp_folder_intelligent(request: RFPProcessingRequest):
    """Intelligently process an RFP folder with document prioritization and token management"""
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
            print(f"📄 Raw content text: {list_result.content[0].text}")
            files_data = json.loads(list_result.content[0].text)
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
        primary_docs = document_prioritizer.get_primary_documents(prioritized_docs, max_count=3)
        secondary_docs = document_prioritizer.get_secondary_documents(prioritized_docs, max_count=2)
        
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

### 📄 **Document: {doc.filename}**

#### 📹 **What is This About?**
> A 3–5 sentence **plain-English overview** of the opportunity. Include:
- Who issued it (organization)
- What they need / are requesting
- Why (the business problem or goal)
- Type of response expected (proposal, quote, info)

---

#### 🧩 **Key Opportunity Details**
List all of the following **if available** in the document:
- **Submission Deadline:** [Date + Time - be specific]
- **Project Start/End Dates:** [Exact dates, contract terms, renewal options]
- **Estimated Value / Budget:** [If stated, include any budget ranges]
- **Response Format:** (e.g., PDF proposal, online portal, pricing form, etc.)
- **Delivery Location(s):** [City, Region, Remote, on-site requirements]
- **Eligibility Requirements:** (Certifications, licenses, location limits, insurance requirements, background checks)
- **Scope Summary:** (Detailed bullet points covering ALL services, technologies, and deliverables mentioned)
- **Specific Technologies:** (List all software, hardware, systems mentioned by name)
- **Insurance Requirements:** (Any specific coverage amounts or types required)
- **Staff Requirements:** (Background checks, certifications, experience levels)

---

#### 📊 **Evaluation Criteria**
How will responses be scored or selected? Include weighting if provided (e.g., 40% price, 30% experience).

---

#### ⚠️ **Notable Risks or Challenges**
Mention anything that could pose a red flag or require clarification (tight timeline, vague scope, legal constraints, strict eligibility, insurance requirements, background checks, geographic requirements, pricing constraints).

---

#### 💡 **Potential Opportunities or Differentiators**
Highlight anything that could give a competitive edge or present upsell/cross-sell opportunities (e.g., optional services, innovation clauses, incumbent fatigue, contract extensions, additional work potential, technology upgrades).

---

#### 📞 **Contact & Submission Info**
- **Primary Contact:** Name, title, email, phone (if listed)
- **Submission Instructions:** Portal, email, physical, etc.

⚠️ **Only summarize what is clearly and explicitly stated. Never guess or infer.**

**CRITICAL:** You must extract ALL specific details from the document including:
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
                    result = await run_mcp_query(
                        analysis_prompt,
                        enabled_tools=['sharepoint_list_files', 'extract_pdf_text'],
                        model=token_manager.get_recommended_model(analysis_prompt, 1000),
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
        
        # Step 5: Generate comprehensive Go/No-Go analysis using the BroadAxis-AI framework
        go_no_go_prompt = f"""You are BroadAxis-AI, an assistant trained to evaluate whether BroadAxis should pursue an RFP, RFQ, or RFI opportunity. The user has uploaded opportunity documents to SharePoint, and you have already analyzed them. Now perform a structured **Go/No-Go analysis** using the following steps:

---

# 🚀 **Intelligent RFP Processing Complete**

## 📊 **Processing Summary**
| **Metric** | **Value** |
|------------|-----------|
| **📁 Folder** | `{folder_path}` |
| **📄 Total Documents** | `{len(documents)}` |
| **🎯 Primary Documents** | `{len(primary_docs)}` |
| **📋 Secondary Documents** | `{len(secondary_docs)}` |
| **⚡ Processing Strategy** | Primary documents only, chunking for large files |

---

## 📄 **Document Analysis Results**

{chr(10).join([f"### 📄 **{doc['filename']}**\n\n{doc['analysis']}\n\n---\n" for doc in processed_documents])}

---

## 🧠 **Comprehensive Go/No-Go Analysis**

### 🔍 **Step 1: RFP Requirements Review**
> **CRITICAL:** Base your analysis ONLY on the document analysis results provided above. Do not hallucinate or make assumptions.

**Key Requirements Identified:**
- Highlight the most critical needs and evaluation criteria from the document analysis above
- Extract key deliverables, timeline, and scope requirements from the actual documents
- Identify any special compliance or certification requirements mentioned in the documents

### 🔎 **Step 2: Internal Knowledge Research**
Use `Broadaxis_knowledge_search` to research:
- Relevant past projects and similar work experience
- Proof of experience in the specific domain/industry
- Known strengths or capability gaps for this type of opportunity
- Government/public sector experience if applicable
- Geographic presence and local capabilities

**🎯 BroadAxis Strengths Identified:**
[Based on knowledge search results, list key capabilities and experience]

### ⚖️ **Step 3: Capability Alignment Assessment**
- Estimate percentage match (e.g., "BroadAxis meets ~85% of the requirements")
- Note any missing capabilities or unclear requirements
- Identify areas where BroadAxis has strong competitive advantages
- Highlight any capability gaps that need to be addressed

### 👥 **Step 4: Resource Requirements Analysis**
- Are there any specialized skills, timelines, or staffing needs?
- Does BroadAxis have the necessary team or partners?
- Analyze proposal deadline and project timeline constraints
- Assess current team and capability readiness

### 🏆 **Step 5: Competitive Positioning Evaluation**
- Based on known experience and domain, would BroadAxis be competitive?
- Identify competitive advantages (local presence, certifications, experience, technology)
- Note potential competitive challenges or weaknesses

---

## 🚦 **FINAL RECOMMENDATION**

### **Decision: [GO / NO-GO / CONDITIONAL GO]**

**📝 Rationale:**
[Clear explanation of the decision with supporting evidence from knowledge search]

**🎯 Confidence Level:** [High / Medium / Low] - [Brief explanation of confidence factors]

---

## 📋 **Action Plan** *(if GO/CONDITIONAL GO)*

### **⚡ Immediate Actions (Next 7 Days):**
1. [Specific capability assessment needed]
2. [Experience documentation required]
3. [Strategic positioning tasks]

### **📝 RFP Response Preparation (Week 2):**
1. [Required forms completion]
2. [Technical response development]
3. [Final submission preparation]

**Risk Mitigation Strategies:**
[Specific strategies to address identified risks]

**Success Probability:** [XX%] with proper preparation, versus [XX%] without focused effort.

## ⚠️ **Important Guidelines:**
- Use only verified internal information (via Broadaxis_knowledge_search) and the uploaded documents
- **CRITICAL:** Base your analysis ONLY on the document analysis results provided above
- Do not guess, hallucinate, or make assumptions about RFP requirements
- If information is missing, clearly state what else is needed for a confident decision
- If your recommendation is a GO, list down the specific tasks the user needs to complete for RFP submission

**ANTI-HALLUCINATION RULE:** Only reference requirements, technologies, and details that are explicitly mentioned in the document analysis results above. Do not mention BI, data warehousing, or other technologies unless they appear in the actual documents.

Provide your analysis in the exact format above. Be thorough, data-driven, and actionable.

**🔧 Tools Used:** Broadaxis_knowledge_search"""
        
        summary_result = await run_mcp_query(
            go_no_go_prompt,
            enabled_tools=['Broadaxis_knowledge_search'],
            model=token_manager.get_recommended_model(go_no_go_prompt, 2000),
            session_id=session_id
        )
        
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
        
        # Format the response with proper markdown structure and spacing
        formatted_response = f"""# 🚀 **Intelligent RFP Processing Complete**

## 📊 **Document Classification Results**

### 📘 **Primary Documents (RFP/RFQ/RFI Content)**

{chr(10).join([f"**{doc['filename']}**\n\n{doc['analysis']}\n" for doc in processed_documents if doc.get('priority') == 'primary'] or ['No primary documents found.'])}

### 📄 **Secondary Documents (Supporting Information)**

{chr(10).join([f"**{doc['filename']}**\n\n{doc['analysis']}\n" for doc in processed_documents if doc.get('priority') == 'secondary'] or ['No secondary documents found.'])}

### 📋 **Other Documents (Reference/Supporting)**

{chr(10).join([f"**{doc['filename']}**\n\n{doc['analysis']}\n" for doc in processed_documents if doc.get('priority') not in ['primary', 'secondary']] or ['No other documents found.'])}

---

## 📋 **Summary**

**Primary Documents:** {len([doc for doc in processed_documents if doc.get('priority') == 'primary'])} files
**Secondary Documents:** {len([doc for doc in processed_documents if doc.get('priority') == 'secondary'])} files  
**Other Documents:** {len([doc for doc in processed_documents if doc.get('priority') not in ['primary', 'secondary']])} files
**Total Documents Processed:** {len(processed_documents)}
**Total Tokens Used:** {total_tokens_used}

---

## 🧠 **Comprehensive Go/No-Go Analysis**

{summary_result.get("response", "")}

---

## 📈 **Session Information**
- **Session ID:** {session_id}
- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

