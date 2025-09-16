"""
WebSocket API for BroadAxis RFP/RFQ Management Platform
"""

import asyncio
import json
import time
from typing import List
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from error_handler import BroadAxisError, ExternalAPIError, error_handler

# Import the MCP interface and manager from mcp_interface.py
from mcp_interface import mcp_interface, run_mcp_query

# Import session manager (optional for now)
try:
    from session_manager import session_manager
    SESSION_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Session manager not available: {e}")
    SESSION_MANAGER_AVAILABLE = False
    session_manager = None

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

# Create manager instance
manager = ConnectionManager()

def is_trading_query(query: str, enabled_tools: List[str] = None) -> bool:
    """Detect if this is a trading-related query that should use Opus 4.1"""
    if not enabled_tools:
        return False
    
    # Check if trading tools are enabled
    trading_tools = ["batch_earnings_analysis", "web_search_tool"]
    has_trading_tools = any(tool in enabled_tools for tool in trading_tools)
    
    if not has_trading_tools:
        return False
    
    # Check for trading-related keywords
    query_lower = query.lower()
    trading_keywords = [
        'stock', 'stocks', 'trading', 'trade', 'market', 'price', 'quote',
        'analyze', 'analysis', 'buy', 'sell', 'portfolio', 'investment',
        'earnings', 'dividend', 'volatility', 'options', 'crypto', 'bitcoin',
        'nasdaq', 'nyse', 'sp500', 'dow', 'apple', 'microsoft', 'google',
        'tesla', 'amazon', 'meta', 'nvidia', 'amd', 'intel'
    ]
    
    return any(keyword in query_lower for keyword in trading_keywords)

async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication."""
    session_id = None  # Will be set from message or created new
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
                
                # Extract or create session_id
                session_id = message_data.get("session_id")
                print(f"🔍 Received session_id from frontend: {session_id}")
                
                # Check if session_id is temporary (starts with 'temp_') or doesn't exist
                if not session_id or (session_id and session_id.startswith('temp_')):
                    if SESSION_MANAGER_AVAILABLE:
                        session_id = await session_manager.create_session()
                        print(f"🆕 Created new Redis session: {session_id}")
                    else:
                        session_id = f"ws_{id(websocket)}_{int(time.time())}"
                        print(f"🆕 Created fallback session: {session_id}")
                else:
                    # Session ID provided and not temporary - verify it exists
                    if SESSION_MANAGER_AVAILABLE:
                        existing_session = await session_manager.get_session(session_id)
                        if not existing_session:
                            print(f"⚠️ Provided session_id {session_id} not found, creating new one")
                            session_id = await session_manager.create_session()
                            print(f"🆕 Created replacement Redis session: {session_id}")
                        else:
                            print(f"✅ Using existing session: {session_id}")
                    else:
                        print(f"✅ Using provided session_id: {session_id}")
                
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
                            "session_id": session_id,
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
                    # Get conversation history for context (if session manager available)
                    conversation_history = []
                    context_prompt = ""
                    
                    if SESSION_MANAGER_AVAILABLE:
                        # First, get existing conversation history
                        conversation_history = await session_manager.get_conversation_context(session_id)
                        
                        # Include conversation history in AI prompt
                        if conversation_history:
                            context_prompt = "Previous conversation:\n"
                            for msg in conversation_history[-10:]:  # Last 10 messages
                                context_prompt += f"{msg['role']}: {msg['content']}\n"
                            context_prompt += "\nCurrent question: "
                            print(f"🧠 Including {len(conversation_history)} previous messages in context")
                            print(f"📝 Context prompt: {context_prompt[:200]}...")
                        else:
                            print(f"🧠 No previous conversation history for session {session_id}")
                        
                        # Add user message to history AFTER getting context
                        user_message = {
                            "role": "user",
                            "content": query,
                            "timestamp": datetime.now().isoformat()
                        }
                        await session_manager.add_message(session_id, user_message)
                    
                    # Process query with enabled tools (guarded per-socket)
                    full_prompt = context_prompt + query
                    print(f"🤖 Sending to AI: {full_prompt[:300]}...")
                    
                    # Check if this is a trading query and use Claude Sonnet 3.7
                    if is_trading_query(query, enabled_tools):
                        print(f"🚀 Detected trading query - using Claude Sonnet 3.7 (cost-optimized)")
                        selected_model = "claude-3-7-sonnet-20250219"
                    else:
                        print(f"📝 Regular query - using standard model")
                        selected_model = model
                    
                    async with websocket.__gate:
                        result = await run_mcp_query(
                            full_prompt, enabled_tools, selected_model, session_id
                        )
                    
                    if not isinstance(result, dict) or "response" not in result:
                        raise ExternalAPIError("Invalid response from query processor")
                    
                    # Add AI response to history (if session manager available)
                    if SESSION_MANAGER_AVAILABLE:
                        ai_message = {
                            "role": "assistant",
                            "content": result["response"],
                            "timestamp": datetime.now().isoformat()
                        }
                        await session_manager.add_message(session_id, ai_message)
                    
                    # Send response with session_id and token details
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "response",
                            "message": result["response"],
                            "session_id": session_id,
                            "status": "success",
                            "tokens_used": result.get("tokens_used", 0),
                            "input_tokens": result.get("input_tokens", 0),
                            "output_tokens": result.get("output_tokens", 0),
                            "model_used": result.get("model_used", "unknown"),
                            "request_id": result.get("request_id", "unknown")
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

async def websocket_rfp_processing(websocket: WebSocket):
    """WebSocket endpoint for real-time RFP processing with streaming updates."""
    session_id = None
    
    try:
        await manager.connect(websocket)
        
        while True:
            try:
                # Receive message from client with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)
                
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
                    if message_type == "ping":
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "pong",
                                "timestamp": datetime.now().isoformat()
                            }),
                            websocket
                        )
                    continue
                
                # Handle RFP processing requests
                if message_type == "rfp_processing":
                    folder_path = message_data.get("folder_path", "")
                    session_id = message_data.get("session_id", f"rfp_{int(time.time())}")
                    
                    if not folder_path:
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "error",
                                "message": "Folder path is required for RFP processing",
                                "status": "error"
                            }),
                            websocket
                        )
                        continue
                    
                    # Start RFP processing with streaming
                    await process_rfp_with_streaming(websocket, folder_path, session_id)
                    
                else:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": f"Unknown message type: {message_type}",
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
                error_handler.log_error(e, {'session_id': session_id, 'operation': 'websocket_rfp_processing'})
                break

    except WebSocketDisconnect:
        error_handler.logger.info(f"WebSocket RFP processing disconnected: {session_id}")

    except Exception as e:
        error_handler.log_error(e, {'session_id': session_id, 'operation': 'websocket_rfp_connection'})

    finally:
        manager.disconnect(websocket)

async def process_rfp_with_streaming(websocket: WebSocket, folder_path: str, session_id: str):
    """Process RFP with real-time streaming updates."""
    try:
        # Import here to avoid circular imports
        from api import process_rfp_folder_intelligent_streaming
        
        # Send initial status
        await manager.send_personal_message(
            json.dumps({
                "type": "rfp_status",
                "message": "🚀 Starting Intelligent RFP Processing...",
                "status": "info",
                "step": "initialization",
                "progress": 0
            }),
            websocket
        )
        
        # Process RFP with streaming updates
        await process_rfp_folder_intelligent_streaming(websocket, folder_path, session_id)
        
    except Exception as e:
        error_handler.log_error(e, {'session_id': session_id, 'folder_path': folder_path})
        await manager.send_personal_message(
            json.dumps({
                "type": "error",
                "message": f"Error during RFP processing: {str(e)}",
                "status": "error"
            }),
            websocket
        )
