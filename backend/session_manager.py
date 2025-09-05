"""
Redis Session Manager for BroadAxis RFP/RFQ Management Platform
"""

import json
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import redis.asyncio as redis
from error_handler import BroadAxisError, ExternalAPIError, error_handler


class RedisSessionManager:
    def __init__(self, redis_url: str = None):
        # Default to your Redis Cloud instance
        self.redis_url = redis_url or "redis://default:Z2lpuSzdHFo6Di14SpHHry4ntcp9MGzv@redis-13430.c85.us-east-1-2.ec2.redns.redis-cloud.com:13430"
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
        
        # Store with 24-hour TTL for testing (can be extended)
        await self.redis.setex(f"session:{session_id}", 86400, json.dumps(session_data))
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
        if not self.redis:
            await self.connect()
            
        session = await self.get_session(session_id)
        if session:
            session["messages"].append(message)
            session["updated_at"] = datetime.now().isoformat()
            
            # Keep only last 100 messages to prevent memory issues
            if len(session["messages"]) > 100:
                session["messages"] = session["messages"][-100:]
                print(f"⚠️ Truncated session {session_id} to last 100 messages")
            
            await self.redis.setex(f"session:{session_id}", 86400, json.dumps(session))
            print(f"✅ Added message to session {session_id}, total: {len(session['messages'])}")
        else:
            print(f"❌ Cannot add message to non-existent session: {session_id}")
            
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
            await self.redis.setex(f"session:{session_id}", 86400, json.dumps(session))
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
            await self.redis.setex(f"session:{session_id}", 86400, json.dumps(session))
            print(f"✅ Stored RFP analysis for session {session_id}")
    
    async def get_rfp_analyses(self, session_id: str) -> List[Dict]:
        """Get stored RFP analyses for a session"""
        session = await self.get_session(session_id)
        if session:
            return session.get("rfp_analyses", [])
        return []
    
    async def store_document_summary(self, session_id: str, document_path: str, summary: Dict):
        """Store document summary for reuse"""
        if not self.redis:
            await self.connect()
            
        session = await self.get_session(session_id)
        if session:
            if "document_summaries" not in session:
                session["document_summaries"] = {}
            
            session["document_summaries"][document_path] = {
                "timestamp": datetime.now().isoformat(),
                "summary": summary
            }
            
            session["updated_at"] = datetime.now().isoformat()
            await self.redis.setex(f"session:{session_id}", 86400, json.dumps(session))
            print(f"✅ Stored document summary for {document_path}")
    
    async def get_document_summary(self, session_id: str, document_path: str) -> Optional[Dict]:
        """Get stored document summary"""
        session = await self.get_session(session_id)
        if session:
            summaries = session.get("document_summaries", {})
            return summaries.get(document_path)
        return None
    
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
            await self.redis.setex(f"session:{session_id}", 86400, json.dumps(session))
            print(f"✅ Stored capability match for {requirement}")
    
    async def get_capability_match(self, session_id: str, requirement: str) -> Optional[Dict]:
        """Get stored capability match"""
        session = await self.get_session(session_id)
        if session:
            matches = session.get("capability_matches", {})
            return matches.get(requirement)
        return None
            
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
        print(f"ℹ️ Skipping cleanup for testing - sessions will persist for 24 hours")
        return 0


# Create global instance
session_manager = RedisSessionManager()
