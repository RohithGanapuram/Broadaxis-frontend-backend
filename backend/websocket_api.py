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
                            query, enabled_tools, model, session_id, websocket, manager.send_personal_message
                        )
                    
                    if not isinstance(result, dict) or "response" not in result:
                        raise ExternalAPIError("Invalid response from query processor")
                    
                    # Send response
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "response",
                            "message": result["response"],
                            "status": "success",
                            "tokens_used": result.get("tokens_used", 0)
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
