"""
FastAPI Backend for BroadAxis RFP/RFQ Management Platform
"""

import asyncio
import json
import os
import time
import base64
from typing import Dict, List, Optional
from datetime import datetime

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

# Import WebSocket functionality
from websocket_api import websocket_chat

# Import API routers
from email_api import email_router
from sharepoint_api import sharepoint_router



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

# Session file storage
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)