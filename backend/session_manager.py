"""
Redis Session Manager for BroadAxis RFP/RFQ Management Platform
"""

import json
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import redis.asyncio as redis
from error_handler import BroadAxisError, ExternalAPIError, error_handler


class RedisSessionManager:
    def __init__(self, redis_url: str = None):
        # Get Redis URL from environment variable for security
        self.redis_url = redis_url or os.getenv('REDIS_URL')
        if not self.redis_url:
            raise ValueError("REDIS_URL environment variable is required. Please set it in your .env file.")
        self.redis = None
        
    async def connect(self):
        """Initialize Redis connection"""
        try:
            self.redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.redis.ping()
            print("✅ Redis connection established successfully")
        except Exception as e:
            error_handler.log_error(e, {'operation': 'redis_connect'})
            print(f"❌ Redis connection failed: {e}")
            raise ExternalAPIError("Failed to connect to Redis", {'original_error': str(e)})
        
    async def create_session(self, user_id: str = None) -> str:
        """Create new session and return session ID"""
        if not self.redis:
            await self.connect()
            
        session_id = str(uuid.uuid4())
        session_data = {
            "id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "files": [],
            "context": {}
        }
        
        # Store with 2-day TTL (172800 seconds = 48 hours)
        await self.redis.setex(f"session:{session_id}", 172800, json.dumps(session_data))
        print(f"✅ Created session: {session_id}")
        return session_id
        
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve session data"""
        if not self.redis:
            await self.connect()
            
        data = await self.redis.get(f"session:{session_id}")
        if data:
            session = json.loads(data)
            print(f"✅ Retrieved session: {session_id} with {len(session.get('messages', []))} messages")
            return session
        else:
            print(f"⚠️ Session not found: {session_id}")
            return None
        
    async def add_message(self, session_id: str, message: Dict):
        """Add message to session history"""
        try:
            if not self.redis:
                await self.connect()
                
            # Validate session_id
            if not session_id or not isinstance(session_id, str):
                print(f"❌ Invalid session_id: {session_id}")
                return False
                
            # Validate message
            if not message or not isinstance(message, dict):
                print(f"❌ Invalid message format: {message}")
                return False
                
            session = await self.get_session(session_id)
            if session:
                session["messages"].append(message)
                session["updated_at"] = datetime.now().isoformat()
                
                # Generate title from first user message if no title exists or title is generic
                if not session.get("title") or session.get("title", "").startswith("New Chat") or session.get("title") == "Untitled Chat":
                    first_user_message = next((msg for msg in session["messages"] if msg.get("role") == "user"), None)
                    if first_user_message and first_user_message.get("content"):
                        title = first_user_message["content"][:30] + ("..." if len(first_user_message["content"]) > 30 else "")
                        session["title"] = title
                        print(f"📝 Generated title for session {session_id}: {title}")
                
                # Keep only last 100 messages to prevent memory issues
                if len(session["messages"]) > 100:
                    session["messages"] = session["messages"][-100:]
                    print(f"⚠️ Truncated session {session_id} to last 100 messages")
                
                await self.redis.setex(f"session:{session_id}", 172800, json.dumps(session))
                print(f"✅ Added message to session {session_id}, total: {len(session['messages'])}")
                return True
            else:
                print(f"❌ Cannot add message to non-existent session: {session_id}")
                return False
        except Exception as e:
            print(f"❌ Error adding message to session {session_id}: {e}")
            return False
            
    async def get_conversation_context(self, session_id: str) -> List[Dict]:
        """Get conversation history for AI context"""
        session = await self.get_session(session_id)
        if session:
            messages = session.get("messages", [])
            print(f"📝 Retrieved {len(messages)} messages for context from session {session_id}")
            return messages
        return []
        
    async def update_context(self, session_id: str, context: Dict):
        """Update session context (files, settings, etc.)"""
        if not self.redis:
            await self.connect()
            
        session = await self.get_session(session_id)
        if session:
            session["context"].update(context)
            session["updated_at"] = datetime.now().isoformat()
            await self.redis.setex(f"session:{session_id}", 172800, json.dumps(session))
            print(f"✅ Updated context for session {session_id}")
    
    async def store_rfp_analysis(self, session_id: str, rfp_analysis: Dict):
        """Store RFP analysis results for reuse"""
        if not self.redis:
            await self.connect()
            
        session = await self.get_session(session_id)
        if session:
            if "rfp_analyses" not in session:
                session["rfp_analyses"] = []
            
            # Add timestamp and analysis
            analysis_with_timestamp = {
                "timestamp": datetime.now().isoformat(),
                "analysis": rfp_analysis
            }
            session["rfp_analyses"].append(analysis_with_timestamp)
            
            # Keep only last 10 analyses to prevent memory issues
            if len(session["rfp_analyses"]) > 10:
                session["rfp_analyses"] = session["rfp_analyses"][-10:]
            
            session["updated_at"] = datetime.now().isoformat()
            await self.redis.setex(f"session:{session_id}", 172800, json.dumps(session))
            print(f"✅ Stored RFP analysis for session {session_id}")
    
    async def get_rfp_analyses(self, session_id: str) -> List[Dict]:
        """Get stored RFP analyses for a session"""
        session = await self.get_session(session_id)
        if session:
            return session.get("rfp_analyses", [])
        return []
    
    async def store_document_summary(self, session_id: str, document_path: str, summary: Dict):
        """Store document summary for reuse - store in both global and session cache"""
        if not self.redis:
            await self.connect()
        
        # Store in global cache (consistent across all users) with 7-day TTL
        global_cache_key = f"document_summary:{document_path}"
        summary_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        }
        await self.redis.setex(global_cache_key, 604800, json.dumps(summary_with_timestamp))  # 7 days
        print(f"✅ Stored global document summary for {document_path}")
        
        # Also store in session cache for backward compatibility
        session = await self.get_session(session_id)
        if session:
            if "document_summaries" not in session:
                session["document_summaries"] = {}
            
            session["document_summaries"][document_path] = summary_with_timestamp
            
            session["updated_at"] = datetime.now().isoformat()
            await self.redis.setex(f"session:{session_id}", 172800, json.dumps(session))
            print(f"✅ Stored session document summary for {document_path}")
    
    async def get_document_summary(self, session_id: str, document_path: str) -> Optional[Dict]:
        """Get stored document summary - check both session and global cache"""
        if not self.redis:
            await self.connect()
        
        # First check global document cache (consistent across all users)
        global_cache_key = f"document_summary:{document_path}"
        global_summary = await self.redis.get(global_cache_key)
        if global_summary:
            print(f"✅ Found global cached summary for {document_path}")
            return json.loads(global_summary)
        
        # Fallback to session-specific cache
        session = await self.get_session(session_id)
        if session:
            summaries = session.get("document_summaries", {})
            return summaries.get(document_path)
        return None
    
    async def clear_document_cache(self, document_path: str = None):
        """Clear document cache - either specific document or all documents"""
        if not self.redis:
            await self.connect()
        
        if document_path:
            # Clear specific document
            global_cache_key = f"document_summary:{document_path}"
            await self.redis.delete(global_cache_key)
            print(f"🗑️ Cleared cache for document: {document_path}")
        else:
            # Clear all document caches
            pattern = "document_summary:*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                print(f"🗑️ Cleared {len(keys)} document caches")
            else:
                print("ℹ️ No document caches found to clear")
    
    async def store_capability_match(self, session_id: str, requirement: str, match_result: Dict):
        """Store capability match results"""
        if not self.redis:
            await self.connect()
            
        session = await self.get_session(session_id)
        if session:
            if "capability_matches" not in session:
                session["capability_matches"] = {}
            
            session["capability_matches"][requirement] = {
                "timestamp": datetime.now().isoformat(),
                "match": match_result
            }
            
            session["updated_at"] = datetime.now().isoformat()
            await self.redis.setex(f"session:{session_id}", 172800, json.dumps(session))
            print(f"✅ Stored capability match for {requirement}")
    
    async def get_capability_match(self, session_id: str, requirement: str) -> Optional[Dict]:
        """Get stored capability match"""
        session = await self.get_session(session_id)
        if session:
            matches = session.get("capability_matches", {})
            return matches.get(requirement)
        return None
            
    async def update_session_title(self, session_id: str, title: str):
        """Update session title"""
        try:
            if not self.redis:
                await self.connect()
                
            session = await self.get_session(session_id)
            if session:
                session["title"] = title
                session["updated_at"] = datetime.now().isoformat()
                await self.redis.setex(f"session:{session_id}", 172800, json.dumps(session))
                print(f"✅ Updated title for session {session_id}: {title}")
                return True
            else:
                print(f"❌ Cannot update title for non-existent session: {session_id}")
                return False
        except Exception as e:
            print(f"❌ Error updating title for session {session_id}: {e}")
            return False

    async def delete_session(self, session_id: str):
        """Delete session"""
        if not self.redis:
            await self.connect()
            
        await self.redis.delete(f"session:{session_id}")
        print(f"🗑️ Deleted session: {session_id}")
        
    async def list_user_sessions(self, user_id: str) -> List[Dict]:
        """List all sessions for a user (simplified for testing)"""
        # For single user testing, return empty list
        # In production, you'd scan for sessions with matching user_id
        return []
        
    async def get_storage_usage(self) -> Dict:
        """Get Redis storage usage information"""
        if not self.redis:
            await self.connect()
            
        info = await self.redis.info('memory')
        return {
            'used_memory': info.get('used_memory', 0),
            'used_memory_peak': info.get('used_memory_peak', 0),
            'max_memory': 30 * 1024 * 1024,  # 30MB
            'max_memory_policy': info.get('maxmemory_policy', 'unknown')
        }
        
    async def cleanup_old_sessions(self, hours: int = 24):
        """Clean up sessions older than specified hours (for testing, we'll skip this)"""
        print(f"ℹ️ Skipping cleanup for testing - sessions will persist for 2 days (48 hours)")
        return 0


# Create global instance
session_manager = RedisSessionManager()
