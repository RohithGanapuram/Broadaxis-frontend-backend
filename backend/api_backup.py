"""
FastAPI Backend for BroadAxis RFP/RFQ Management Platform
This server wraps the existing MCP server and provides REST/WebSocket APIs for the React frontend.
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import traceback
from io import BytesIO
from typing import Dict, List, Optional, Any

import nest_asyncio
import PyPDF2
import docx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI(
    title="BroadAxis API",
    description="FastAPI backend for BroadAxis RFP/RFQ Management Platform",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handling middleware
@app.middleware("http")
async def catch_exceptions_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print(f"Unhandled exception: {e}")
        print(f"Request: {request.method} {request.url}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )

# Location of the MCP server script
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "ba-server", "server.py")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str
    enabled_tools: List[str] = []
    model: str = "claude-3-7-sonnet-20250219"

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

class FileUploadResponse(BaseModel):
    status: str
    filename: str
    size: int
    file_type: str
    message: str

class ErrorResponse(BaseModel):
    error: str
    status: str = "error"

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Global storage for uploaded documents
uploaded_documents = {}

# Global MCP interface for reuse
class MCPInterface:
    def __init__(self):
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[SERVER_SCRIPT],
            env=None,
            cwd=os.path.dirname(SERVER_SCRIPT),
        )
        self.anthropic = None
        self._connection_lock = asyncio.Lock()
        self._tools_cache = None
        self._cache_time = None
        try:
            from anthropic import Anthropic
            self.anthropic = Anthropic()
        except ImportError:
            print("Warning: Anthropic not available")
    
    async def get_tools_and_prompts(self):
        """Return static tools and prompts to avoid MCP server issues"""
        try:
            # Return static tools
            tools = [
                {"name": "Broadaxis_knowledge_search", "description": "Search company knowledge base", "input_schema": {}},
                {"name": "web_search_tool", "description": "Search the web", "input_schema": {}},
                {"name": "generate_pdf_document", "description": "Generate PDF documents", "input_schema": {}},
                {"name": "generate_word_document", "description": "Generate Word documents", "input_schema": {}},
                {"name": "generate_text_file", "description": "Generate text files", "input_schema": {}},
                {"name": "search_papers", "description": "Search academic papers", "input_schema": {}}
            ]
            
            # Return static prompts
            prompts = [
                {"name": "RFP Analysis", "description": "Analyze RFP documents", "arguments": []},
                {"name": "Go/No-Go Decision", "description": "Make Go/No-Go recommendations", "arguments": []},
                {"name": "Company Research", "description": "Research company capabilities", "arguments": []}
            ]
            
            return tools, prompts
        except Exception as e:
            print(f"Error getting tools and prompts: {e}")
            return [], []
    
    async def process_query_with_anthropic(self, query: str, enabled_tools: List[str] = None, model: str = None) -> str:
        """Process query using direct Anthropic API call"""
        if not self.anthropic:
            return "Anthropic API not available"
        
        # For simple queries, skip MCP tools to speed up response
        simple_queries = ['hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye']
        capability_queries = ['what can you do', 'what do you do', 'help', 'capabilities', 'features']
        
        query_lower = query.lower().strip()
        
        if query_lower in simple_queries:
            try:
                doc_count = len(uploaded_documents)
                doc_info = f" I have {doc_count} document(s) available for analysis." if doc_count > 0 else " Upload documents to get started!"
                response = self.anthropic.messages.create(
                    max_tokens=150,
                    model=model or "claude-3-7-sonnet-20250219",
                    messages=[{'role': 'user', 'content': query}]
                )
                return response.content[0].text + doc_info
            except Exception as e:
                doc_count = len(uploaded_documents)
                doc_info = f" I have {doc_count} document(s) available for analysis." if doc_count > 0 else " Upload documents to get started!"
                return f"Hello! I'm BroadAxis-AI, ready to help with your RFP/RFQ needs.{doc_info} How can I assist you today?"
        
        if any(cap in query_lower for cap in capability_queries):
            doc_status = f"\n\n📄 **Currently Available Documents**: {len(uploaded_documents)} document(s) uploaded" if uploaded_documents else "\n\n📄 **No documents uploaded yet** - Upload documents to analyze them!"
            return f"""I'm BroadAxis-AI, your RFP/RFQ management assistant. I can help you with:

🔍 **Document Analysis**: Ask questions about uploaded documents
📊 **Go/No-Go Decisions**: Get recommendations based on company capabilities
🏢 **Company Knowledge**: Search internal expertise and past projects
🌐 **Market Research**: Find external information and industry trends
📄 **Document Generation**: Create PDFs, Word docs, and reports
📚 **Academic Research**: Search relevant papers and publications{doc_status}

Just upload a document and ask me questions about it, or ask about BroadAxis capabilities!"""
        
        # Check if tools are needed based on query content
        needs_tools = self._should_use_tools(query, enabled_tools)
        
        if needs_tools:
            return await self._process_with_tools(query, enabled_tools, model)
        else:
            return await self._process_direct(query, model)
    
    def _should_use_tools(self, query: str, enabled_tools: List[str]) -> bool:
        """Determine if tools are needed for this query"""
        query_lower = query.lower()
        
        # Tool-requiring keywords
        tool_keywords = {
            'broadaxis_knowledge_search': ['company', 'broadaxis', 'team', 'expertise', 'past projects', 'capabilities', 'experience'],
            'web_search_tool': ['search web', 'find online', 'latest news', 'current market', 'research online'],
            'generate_pdf_document': ['create pdf', 'generate pdf', 'pdf document', 'pdf report'],
            'generate_word_document': ['create word', 'generate word', 'word document', 'docx'],
            'generate_text_file': ['create text', 'generate text', 'text file', 'save as text'],
            'search_papers': ['academic', 'research papers', 'arxiv', 'scientific', 'papers']
        }
        
        # Check if any tool keywords are present
        for tool, keywords in tool_keywords.items():
            if tool in enabled_tools and any(keyword in query_lower for keyword in keywords):
                return True
        
        # Check for explicit tool requests
        if any(word in query_lower for word in ['search', 'generate', 'create', 'find information about']):
            return True
            
        return False
    
    async def _process_direct(self, query: str, model: str = None) -> str:
        """Process query with direct Anthropic call (no tools)"""
        try:
            messages = []
            
            # Add document context if available
            if uploaded_documents:
                doc_context = "\n\nUploaded Documents Available:\n"
                for filename, doc_info in uploaded_documents.items():
                    doc_context += f"\n--- {filename} ({doc_info['file_type']}) ---\n"
                    content = doc_info['content'][:4000] + "..." if len(doc_info['content']) > 4000 else doc_info['content']
                    doc_context += content + "\n"
                messages.append({'role': 'system', 'content': f"You have access to these uploaded documents. Use them to answer user questions: {doc_context}"})
            
            messages.append({'role': 'user', 'content': query})
            
            response = self.anthropic.messages.create(
                max_tokens=2000,
                model=model or "claude-3-7-sonnet-20250219",
                system="You are BroadAxis-AI, an RFP/RFQ management assistant. Answer questions directly using available information. Be helpful and conversational.",
                messages=messages
            )
            return response.content[0].text
            
        except Exception as e:
            print(f"Direct Anthropic API error: {e}")
            return f"I'm experiencing technical difficulties. Please try again in a moment."
    
    async def _process_with_tools(self, query: str, enabled_tools: List[str], model: str = None) -> str:
        """Process query with MCP tools when necessary"""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Get available tools
                    tools_response = await session.list_tools()
                    available_tools = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    } for tool in tools_response.tools if tool.name in enabled_tools]
                    
                    messages = []
                    
                    # Add document context
                    if uploaded_documents:
                        doc_context = "\n\nUploaded Documents Available:\n"
                        for filename, doc_info in uploaded_documents.items():
                            doc_context += f"\n--- {filename} ({doc_info['file_type']}) ---\n"
                            content = doc_info['content'][:2000] + "..." if len(doc_info['content']) > 2000 else doc_info['content']
                            doc_context += content + "\n"
                        messages.append({'role': 'system', 'content': f"Available documents: {doc_context}"})
                    
                    messages.append({'role': 'user', 'content': query})
                    
                    response = self.anthropic.messages.create(
                        max_tokens=2000,
                        model=model or "claude-3-7-sonnet-20250219",
                        system="You are BroadAxis-AI. Use tools when needed for company knowledge, web search, or document generation. Be efficient.",
                        tools=available_tools,
                        messages=messages
                    )
                    
                    # Handle tool usage
                    if response.content[0].type == 'text':
                        return response.content[0].text
                    else:
                        # Process tool calls if needed
                        full_response = ""
                        for content in response.content:
                            if content.type == 'text':
                                full_response += content.text
                            elif content.type == 'tool_use':
                                try:
                                    result = await session.call_tool(content.name, arguments=content.input)
                                    full_response += f"\n\n{result.content[0].text if result.content else 'Tool executed successfully'}"
                                except Exception as tool_error:
                                    full_response += f"\n\nTool error: {str(tool_error)}"
                        return full_response
                        
        except Exception as e:
            print(f"Tool processing error: {e}")
            # Fallback to direct call
            return await self._process_direct(query, model)

# Global MCP interface instance
mcp_interface = MCPInterface()

# Start background task on startup
@app.on_event("startup")
async def startup_event():
    """Run startup tasks"""
    try:
        print("✅ BroadAxis API started with static tools and prompts")
    except Exception as e:
        print(f"Startup error: {e}")
        traceback.print_exc()

# Utility function to run MCP server communication
async def run_mcp_query(query: str, enabled_tools: List[str] = None, model: str = None) -> str:
    """Run a query against the MCP server and return the response."""
    return await mcp_interface.process_query_with_anthropic(query, enabled_tools, model)

# Document processing utilities
def extract_text_from_pdf(file_content: bytes) -> tuple[str, int]:
    """Extract text from PDF file."""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text, len(pdf_reader.pages)
    except Exception as e:
        raise ValueError(f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        doc_file = BytesIO(file_content)
        doc = docx.Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        raise ValueError(f"Error reading DOCX: {str(e)}")

def process_uploaded_file_content(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Process uploaded file and extract content."""
    file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
    
    try:
        if file_ext == 'pdf':
            content, page_count = extract_text_from_pdf(file_content)
        elif file_ext in ['docx', 'doc']:
            content = extract_text_from_docx(file_content)
            page_count = None
        elif file_ext in ['txt', 'md']:
            content = file_content.decode('utf-8')
            page_count = None
        else:
            # Try to decode as text
            content = file_content.decode('utf-8', errors='ignore')
            page_count = None
        
        word_count = len(content.split())
        char_count = len(content)
        
        return {
            'status': 'success',
            'filename': filename,
            'size': len(file_content),
            'full_content': content,  # Keep full content for storage
            'word_count': word_count,
            'char_count': char_count,
            'file_type': file_ext,
            'page_count': page_count
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'filename': filename,
            'size': len(file_content),
            'content': f"Error processing file: {str(e)}",
            'word_count': 0,
            'char_count': 0,
            'file_type': file_ext,
            'page_count': None
        }

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
async def health_check():
    """Health check endpoint."""
    try:
        return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}
    except Exception as e:
        print(f"Health check error: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process chat queries through the MCP server."""
    try:
        result = await run_mcp_query(
            query=request.query,
            enabled_tools=request.enabled_tools,
            model=request.model
        )
        return ChatResponse(response=result, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and store files for interaction."""
    try:
        # Read file content
        file_content = await file.read()

        # Process the file
        file_info = process_uploaded_file_content(file_content, file.filename)

        if file_info['status'] == 'error':
            raise HTTPException(status_code=400, detail=file_info['content'])

        # Store document in global storage for chat interaction
        uploaded_documents[file.filename] = {
            'content': file_info['full_content'],
            'size': file_info['size'],
            'file_type': file_info['file_type'],
            'word_count': file_info['word_count'],
            'char_count': file_info['char_count']
        }

        return FileUploadResponse(
            status="success",
            filename=file.filename,
            size=file_info['size'],
            file_type=file_info['file_type'],
            message=f"Document '{file.filename}' uploaded successfully. You can now ask questions about it!"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

@app.get("/api/uploaded-documents")
async def get_uploaded_documents():
    """Get list of uploaded documents."""
    try:
        documents = []
        for filename, doc_info in uploaded_documents.items():
            documents.append({
                'filename': filename,
                'size': doc_info['size'],
                'file_type': doc_info['file_type'],
                'word_count': doc_info['word_count'],
                'char_count': doc_info['char_count']
            })
        return {"documents": documents, "status": "success"}
    except Exception as e:
        return {"documents": [], "status": "error", "error": str(e)}

@app.delete("/api/uploaded-documents/{filename}")
async def remove_uploaded_document(filename: str):
    """Remove an uploaded document from storage."""
    try:
        if filename in uploaded_documents:
            del uploaded_documents[filename]
            return {"status": "success", "message": f"Document '{filename}' removed"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing document: {str(e)}")

@app.post("/api/clear-documents")
async def clear_all_documents():
    """Clear all uploaded documents."""
    try:
        uploaded_documents.clear()
        return {"status": "success", "message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

# Removed caching variables as we're using static data

@app.get("/api/tools")
async def get_available_tools():
    """Get list of available MCP tools."""
    try:
        # Complete list of 12 tools
        all_tools = [
            {"name": "Broadaxis_knowledge_search", "description": "Search company knowledge base", "input_schema": {}},
            {"name": "web_search_tool", "description": "Search the web", "input_schema": {}},
            {"name": "generate_pdf_document", "description": "Generate PDF documents", "input_schema": {}},
            {"name": "generate_word_document", "description": "Generate Word documents", "input_schema": {}},
            {"name": "generate_text_file", "description": "Generate text files", "input_schema": {}},
            {"name": "search_papers", "description": "Search academic papers", "input_schema": {}},
            {"name": "get_weather", "description": "Get current weather information", "input_schema": {}},
            {"name": "get_forecast", "description": "Get weather forecast", "input_schema": {}},
            {"name": "get_alerts", "description": "Get weather alerts", "input_schema": {}},
            {"name": "get_historical_weather", "description": "Get historical weather data", "input_schema": {}},
            {"name": "get_air_quality", "description": "Get air quality information", "input_schema": {}},
            {"name": "get_uv_index", "description": "Get UV index information", "input_schema": {}}
        ]
        
        return {"tools": all_tools, "status": "success", "source": "static"}
    except Exception as e:
        print(f"Error in get_available_tools: {e}")
        return {"tools": [], "status": "error", "error": str(e)}

@app.get("/api/prompts")
async def get_available_prompts():
    """Get list of available MCP prompts."""
    try:
        # Always return static prompts to avoid MCP server issues
        fallback_prompts = [
            {"name": "RFP Analysis", "description": "Analyze RFP documents", "arguments": []},
            {"name": "Go/No-Go Decision", "description": "Make Go/No-Go recommendations", "arguments": []},
            {"name": "Company Research", "description": "Research company capabilities", "arguments": []}
        ]
        
        return {"prompts": fallback_prompts, "status": "success", "source": "static"}
    except Exception as e:
        print(f"Error in get_available_prompts: {e}")
        return {"prompts": [], "status": "error", "error": str(e)}

@app.post("/api/tools/refresh")
async def refresh_tools():
    """Manually refresh tools and prompts from MCP server"""
    return {"status": "success", "message": "Using static tools - refresh not needed"}

class PromptRequest(BaseModel):
    prompt_name: str
    arguments: Dict[str, str] = {}
    model: str = "claude-3-7-sonnet-20250219"

@app.post("/api/prompts/execute")
async def execute_prompt(request: PromptRequest):
    """Execute a specific MCP prompt with arguments"""
    try:
        # For now, just use direct Anthropic call with prompt template
        prompt_templates = {
            "RFP Analysis": "Analyze the following RFP/RFQ document and provide a comprehensive breakdown including executive summary, key requirements, timeline, budget, evaluation criteria, and risks.",
            "Go/No-Go Decision": "Based on the RFP requirements and BroadAxis capabilities, provide a Go/No-Go recommendation with supporting analysis including capability alignment, resource requirements, and competitive positioning.",
            "Company Research": "Research and provide information about BroadAxis company capabilities, past projects, team expertise, and relevant experience for this opportunity."
        }
        
        prompt_text = prompt_templates.get(request.prompt_name, f"Execute the {request.prompt_name} prompt.")
        
        if mcp_interface.anthropic:
            response = mcp_interface.anthropic.messages.create(
                max_tokens=2048,
                model=request.model,
                system="You are BroadAxis-AI, an RFP/RFQ management assistant. Be thorough and professional.",
                messages=[{"role": "user", "content": prompt_text}]
            )
            return {"response": response.content[0].text, "status": "success"}
        else:
            return {"response": "Anthropic API not available", "status": "error"}
            
    except Exception as e:
        return {"response": f"Error executing prompt: {str(e)}", "status": "error"}

# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication."""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            query = message_data.get("query", "")
            enabled_tools = message_data.get("enabled_tools", [])
            model = message_data.get("model", "claude-3-7-sonnet-20250219")

            # Send acknowledgment
            await manager.send_personal_message(
                json.dumps({"type": "status", "message": "Processing your request..."}),
                websocket
            )

            try:
                # Process query with enabled tools
                result = await run_mcp_query(query, enabled_tools, model)

                # Send response
                await manager.send_personal_message(
                    json.dumps({
                        "type": "response",
                        "message": result,
                        "status": "success"
                    }),
                    websocket
                )

            except Exception as e:
                # Send error
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": f"Error processing request: {str(e)}",
                        "status": "error"
                    }),
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# File management endpoints (for generated files)
@app.get("/api/files")
async def list_files():
    """List all generated files"""
    return {"files": [], "status": "success", "count": 0}

@app.get("/api/files/{filename}")
async def download_file(filename: str):
    """Download a specific file"""
    raise HTTPException(status_code=404, detail="File service not available")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Delete a specific file"""
    raise HTTPException(status_code=404, detail="File service not available")

@app.post("/api/files/cleanup")
async def cleanup_files(days_old: int = 7):
    """Clean up old files"""
    return {"status": "error", "message": "File service not available"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
