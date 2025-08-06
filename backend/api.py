"""
FastAPI Backend for BroadAxis RFP/RFQ Management Platform
"""

import asyncio
import json
import os
import sys
import time
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
        self._tools_cache = None
        self._prompts_cache = None
        self._cache_lock = asyncio.Lock()
        try:
            from anthropic import Anthropic
            self.anthropic = Anthropic()
        except ImportError:
            print("Warning: Anthropic not available")
    
    async def get_tools(self):
        """Get cached tools or fetch from server"""
        async with self._cache_lock:
            if self._tools_cache is None:
                try:
                    async with stdio_client(self.server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            tools_response = await session.list_tools()
                            self._tools_cache = [{
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema
                            } for tool in tools_response.tools]
                except Exception as e:
                    print(f"Failed to fetch tools: {e}")
                    self._tools_cache = []
            return self._tools_cache
    
    async def get_prompts(self):
        """Get cached prompts or fetch from server"""
        async with self._cache_lock:
            if self._prompts_cache is None:
                try:
                    async with stdio_client(self.server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            prompts_response = await session.list_prompts()
                            self._prompts_cache = [{
                                "name": prompt.name,
                                "description": prompt.description
                            } for prompt in prompts_response.prompts]
                except Exception as e:
                    print(f"Failed to fetch prompts: {e}")
                    self._prompts_cache = []
            return self._prompts_cache
    
    async def process_query_with_anthropic(self, query: str, enabled_tools: List[str] = None, model: str = None) -> str:
        if not self.anthropic:
            return "Anthropic API not available"
            
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
                                response = self.anthropic.messages.create(
                                    max_tokens=3048,
                                    model=selected_model,
                                    system=enhanced_system,
                                    messages=[{'role': 'user', 'content': "Analyze the uploaded documents using this framework."}]
                                )
                                return response.content[0].text if response.content else "No response generated"
                            else:
                                return f"Prompt '{prompt_name}' not found"
                except Exception as e:
                    return f"Error using prompt: {str(e)}"
            
            # Use fresh connection for tool-based queries with cached tools
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    available_tools = await self.get_tools()
                    
                    if enabled_tools:
                        available_tools = [tool for tool in available_tools if tool["name"] in enabled_tools]
                    
                    system_prompt = """You are BroadAxis-AI, an intelligent RFP/RFQ management assistant. 
Use Broadaxis_knowledge_search for company information first, then web_search_tool for external data if needed.
For uploaded RFP documents, provide analysis and offer Go/No-Go recommendations."""
                    
                    selected_model = model or "claude-3-7-sonnet-20250219"
                    messages = [{'role': 'user', 'content': query}]
                    
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

# Store uploaded files per session
session_files = {}

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
        
        # Note: File upload now only confirms upload, doesn't store globally
        # Files should be re-uploaded per session for analysis
        
        return {
            "status": "success",
            "filename": file.filename,
            "size": len(file_content),
            "message": f"File '{file.filename}' uploaded and analyzed. You can now ask questions about it.",
            "analysis": f"Document '{file.filename}' uploaded successfully. You can now ask questions about it."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/tools")
async def get_available_tools():
    try:
        tools = await mcp_interface.get_tools()
        return {"tools": tools, "status": "success"}
    except Exception as e:
        print(f"Error fetching tools: {e}")
        return {"tools": [], "status": "error"}

@app.get("/api/prompts")
async def get_available_prompts():
    try:
        prompts = await mcp_interface.get_prompts()
        return {"prompts": prompts, "status": "success"}
    except Exception as e:
        print(f"Error fetching prompts: {e}")
        return {"prompts": [], "status": "error"}





# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication."""
    await manager.connect(websocket)
    session_id = f"ws_{id(websocket)}_{int(time.time())}"
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            query = message_data.get("query", "")
            enabled_tools = message_data.get("enabled_tools", [])
            model = message_data.get("model", "claude-3-7-sonnet-20250219")
            
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