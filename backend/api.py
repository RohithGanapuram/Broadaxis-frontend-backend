"""
FastAPI Backend for BroadAxis RFP/RFQ Management Platform
"""

import asyncio
import json
import os
import sys
from typing import Dict, List

import nest_asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

nest_asyncio.apply()

app = FastAPI(title="BroadAxis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ChatRequest(BaseModel):
    query: str
    enabled_tools: List[str] = []
    model: str = "claude-3-7-sonnet-20250219"

class ChatResponse(BaseModel):
    response: str
    status: str = "success"



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
        except Exception:
            self.disconnect(websocket)

manager = ConnectionManager()

class MCPInterface:
    def __init__(self):
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.join(os.path.dirname(__file__), "..", "ba-server", "server.py")],
            env=None,
        )
        self.anthropic = None
        try:
            from anthropic import Anthropic
            self.anthropic = Anthropic()
        except ImportError:
            print("Warning: Anthropic not available")
    
    async def process_query_with_anthropic(self, query: str, enabled_tools: List[str] = None, model: str = None) -> str:
        if not self.anthropic:
            return "Anthropic API not available"
            
        try:
            # Check if there are uploaded files to include
            file_context = ""
            has_files = bool(uploaded_files_content)
            
            if has_files:
                # Determine content strategy based on query type
                is_summary_query = any(word in query.lower() for word in ['summary', 'summarize', 'about', 'analyze', 'what is', 'overview'])
                is_specific_query = any(word in query.lower() for word in ['find', 'search', 'where', 'how', 'when', 'who', 'specific', 'detail'])
                
                file_context = "\n\nUploaded Documents:\n"
                for filename, file_data in uploaded_files_content.items():
                    if is_summary_query:
                        # Use summary content for overview questions
                        content = file_data['summary_content']
                    elif is_specific_query:
                        # Use full content for detailed questions
                        content = file_data['full_content'][:15000]  # Limit to 15k for specific queries
                    else:
                        # Default to summary content
                        content = file_data['summary_content']
                    
                    file_context += f"\n--- {filename} ---\n{content}\n"
            
            enhanced_query = query + file_context
            
            # Handle MCP prompt invocations
            if query.startswith('PROMPT:'):
                prompt_name = query[7:]  # Remove 'PROMPT:' prefix
                try:
                    async with stdio_client(self.server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            
                            # Get the prompt template
                            prompts_response = await session.list_prompts()
                            target_prompt = None
                            for prompt in prompts_response.prompts:
                                if prompt.name == prompt_name:
                                    target_prompt = prompt
                                    break
                            
                            if target_prompt:
                                # Get prompt content
                                prompt_result = await session.get_prompt(target_prompt.name, arguments={})
                                prompt_content = prompt_result.messages[0].content.text if prompt_result.messages else ""
                                
                                # Use the prompt content as system message with formatting instructions
                                enhanced_system = prompt_content + "\n\nIMPORTANT: Format your response with clear headings, bullet points, and proper spacing for readability. Use markdown formatting with ## for main sections, ### for subsections, and bullet points for lists."
                                
                                selected_model = model or "claude-3-7-sonnet-20250219"
                                response = self.anthropic.messages.create(
                                    max_tokens=3048,
                                    model=selected_model,
                                    system=enhanced_system,
                                    messages=[{'role': 'user', 'content': f"Analyze the uploaded documents using this framework.{file_context}"}]
                                )
                                return response.content[0].text if response.content else "No response generated"
                            else:
                                return f"Prompt '{prompt_name}' not found"
                except Exception as e:
                    return f"Error using prompt: {str(e)}"
            
            # For document analysis queries, use direct API (faster)
            if has_files and any(word in query.lower() for word in ['summary', 'summarize', 'about', 'analyze', 'what is', 'overview']):
                selected_model = model or "claude-3-7-sonnet-20250219"
                response = self.anthropic.messages.create(
                    max_tokens=2048,
                    model=selected_model,
                    system="You are BroadAxis-AI, an intelligent RFP/RFQ management assistant. Analyze the uploaded documents and provide clear, concise summaries. Format your response with proper headings (##), bullet points, and clear sections for easy reading.",
                    messages=[{'role': 'user', 'content': enhanced_query}]
                )
                return response.content[0].text if response.content else "No response generated"
            
            # Use MCP server for tool-based queries
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    tools_response = await session.list_tools()
                    available_tools = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    } for tool in tools_response.tools]
                    
                    if enabled_tools:
                        available_tools = [tool for tool in available_tools if tool["name"] in enabled_tools]
                    
                    system_prompt = """You are BroadAxis-AI, an intelligent RFP/RFQ management assistant. 
Use Broadaxis_knowledge_search for company information first, then web_search_tool for external data if needed.
For uploaded RFP documents, provide analysis and offer Go/No-Go recommendations.
When users ask about uploaded documents, analyze the content provided in the user message."""
                    
                    selected_model = model or "claude-3-7-sonnet-20250219"
                    messages = [{'role': 'user', 'content': enhanced_query}]
                    
                    response = self.anthropic.messages.create(
                        max_tokens=4024,
                        model=selected_model,
                        system=system_prompt,
                        tools=available_tools,
                        messages=messages
                    )
                    
                    full_response = ""
                    tools_used = []
                    process_query = True
                    
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
                                
                                response = self.anthropic.messages.create(
                                    max_tokens=2024,
                                    model=selected_model,
                                    system=system_prompt,
                                    tools=available_tools,
                                    messages=messages
                                )
                                
                                if len(response.content) == 1 and response.content[0].type == "text":
                                    full_response += response.content[0].text
                                    process_query = False
                    
                    # Add tools used information to response
                    if tools_used:
                        tools_info = "\n\n---\n🔧 **Tools Used:** " + ", ".join(set(tools_used))
                        full_response += tools_info
                    
                    return full_response
                    
        except Exception as e:
            return f"Error: {str(e)}"

mcp_interface = MCPInterface()

async def run_mcp_query(query: str, enabled_tools: List[str] = None, model: str = None) -> str:
    return await mcp_interface.process_query_with_anthropic(query, enabled_tools, model)

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
    import time
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "message": "BroadAxis API is running",
        "version": "1.0.0"
    }

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

# Store uploaded files temporarily
uploaded_files_content = {}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        
        # Store file content for chat analysis
        text_content = ""
        if file.filename.lower().endswith('.pdf'):
            # Basic PDF text extraction
            try:
                import PyPDF2
                from io import BytesIO
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                text_content = "\n".join([page.extract_text() for page in pdf_reader.pages])
            except:
                text_content = "PDF content could not be extracted"
        else:
            # For text files
            text_content = file_content.decode('utf-8', errors='ignore')
        
        # Store full content for comprehensive analysis
        uploaded_files_content[file.filename] = {
            "full_content": text_content,  # Store complete content
            "summary_content": text_content[:8000],  # First 8k for summaries
            "filename": file.filename,
            "size": len(file_content)
        }
        
        return {
            "status": "success",
            "filename": file.filename,
            "size": len(file_content),
            "message": f"File '{file.filename}' uploaded and analyzed. You can now ask questions about it.",
            "analysis": f"Document '{file.filename}' has been processed and is ready for analysis. Ask me questions about its content."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/tools")
async def get_available_tools():
    try:
        async with stdio_client(mcp_interface.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                tools = [{"name": tool.name, "description": tool.description} for tool in tools_response.tools]
                return {"tools": tools, "status": "success"}
    except Exception as e:
        print(f"Error fetching tools: {e}")
        # Fallback to static tools if MCP server fails
        static_tools = [
            {"name": "sum", "description": "Add two numbers"},
            {"name": "Broadaxis_knowledge_search", "description": "Search company knowledge base"},
            {"name": "web_search_tool", "description": "Search the web using Tavily"},
            {"name": "generate_pdf_document", "description": "Generate PDF documents"},
            {"name": "generate_word_document", "description": "Generate Word documents"},
            {"name": "generate_text_file", "description": "Generate text files"},
            {"name": "search_papers", "description": "Search academic papers on arXiv"},
            {"name": "get_forecast", "description": "Get weather forecast"},
            {"name": "get_alerts", "description": "Get weather alerts"},
            {"name": "list_generated_files", "description": "List generated files"}
        ]
        return {"tools": static_tools, "status": "fallback"}

@app.get("/api/prompts")
async def get_available_prompts():
    try:
        async with stdio_client(mcp_interface.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                prompts_response = await session.list_prompts()
                prompts = [{"name": prompt.name, "description": prompt.description} for prompt in prompts_response.prompts]
                return {"prompts": prompts, "status": "success"}
    except Exception as e:
        print(f"Error fetching prompts: {e}")
        # Fallback to static prompts if MCP server fails
        static_prompts = [
            {"name": "Step-2: Executive Summary", "description": "Generate executive summary of RFP documents"},
            {"name": "Step-3: Go/No-Go Recommendation", "description": "Provide Go/No-Go analysis"},
            {"name": "Step-4: Generate Proposal", "description": "Generate capability statement"}
        ]
        return {"prompts": static_prompts, "status": "fallback"}





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
                # Send error only if connection is still active
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
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/api/files")
async def list_files():
    # Skip MCP server and use direct file listing for now
    print("Using direct file listing (MCP server bypassed)")
    
    # Direct file listing
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
        
        return {"files": files, "status": "success"}
    except Exception as e:
        print(f"Direct file listing failed: {e}")
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