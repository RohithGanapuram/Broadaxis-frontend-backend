"""
Token Management System for BroadAxis RFP Platform
Handles token budgeting, tracking, and rate limiting
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ModelType(Enum):
    HAIKU = "claude-3-haiku-20240307"
    SONNET = "claude-3-5-sonnet-20241022"
    SONNET_7 = "claude-3-7-sonnet-20250219"
    OPUS = "claude-3-opus-20240229"

class TaskComplexity(Enum):
    SIMPLE = "simple"      # Document prioritization, basic classification
    MEDIUM = "medium"      # Document analysis, summarization
    COMPLEX = "complex"    # Go/No-Go decisions, strategic analysis

@dataclass
class TokenBudget:
    """Token budget configuration for each model"""
    hourly_limit: int
    daily_limit: int
    current_usage: int = 0
    last_reset: datetime = None
    request_count: int = 0
    max_concurrent_requests: int = 3
    
    def __post_init__(self):
        if self.last_reset is None:
            self.last_reset = datetime.now()

@dataclass
class TokenUsage:
    """Record of token usage for a request"""
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: datetime
    request_id: str
    task_type: str
    session_id: str

@dataclass
class RateLimitInfo:
    """Rate limiting information"""
    requests_per_minute: int
    requests_per_hour: int
    current_requests: deque
    last_cleanup: datetime

class TokenManager:
    """Comprehensive token and rate limit management"""
    
    def __init__(self):
        # Token budgets for each model (conservative limits)
        self.budgets = {
            ModelType.HAIKU.value: TokenBudget(
                hourly_limit=50000,    # 50k tokens/hour
                daily_limit=500000,    # 500k tokens/day
                max_concurrent_requests=5
            ),
            ModelType.SONNET.value: TokenBudget(
                hourly_limit=25000,    # 25k tokens/hour
                daily_limit=250000,    # 250k tokens/day
                max_concurrent_requests=3
            ),
            ModelType.SONNET_7.value: TokenBudget(
                hourly_limit=25000,    # 25k tokens/hour
                daily_limit=250000,    # 250k tokens/day
                max_concurrent_requests=3
            ),
            ModelType.OPUS.value: TokenBudget(
                hourly_limit=10000,    # 10k tokens/hour
                daily_limit=100000,    # 100k tokens/day
                max_concurrent_requests=2
            )
        }
        
        # Rate limiting (more permissive for testing)
        self.rate_limits = {
            ModelType.HAIKU.value: RateLimitInfo(
                requests_per_minute=60,  # Increased from 30
                requests_per_hour=2000,  # Increased from 1000
                current_requests=deque(),
                last_cleanup=datetime.now()
            ),
            ModelType.SONNET.value: RateLimitInfo(
                requests_per_minute=40,  # Increased from 20
                requests_per_hour=1000,  # Increased from 500
                current_requests=deque(),
                last_cleanup=datetime.now()
            ),
            ModelType.SONNET_7.value: RateLimitInfo(
                requests_per_minute=40,  # Same as SONNET
                requests_per_hour=1000,  # Same as SONNET
                current_requests=deque(),
                last_cleanup=datetime.now()
            ),
            ModelType.OPUS.value: RateLimitInfo(
                requests_per_minute=20,  # Increased from 10
                requests_per_hour=400,   # Increased from 200
                current_requests=deque(),
                last_cleanup=datetime.now()
            )
        }
        
        # Usage tracking
        self.usage_history: List[TokenUsage] = []
        self.session_usage: Dict[str, List[TokenUsage]] = defaultdict(list)
        
        # Concurrent request tracking
        self.active_requests: Dict[str, int] = defaultdict(int)
        
        # Model selection strategy
        self.model_strategy = {
            TaskComplexity.SIMPLE: ModelType.HAIKU,
            TaskComplexity.MEDIUM: ModelType.SONNET,
            TaskComplexity.COMPLEX: ModelType.OPUS
        }
        
        # Cleanup task will be started when first needed
        self._cleanup_task = None
    
    def _start_cleanup_task(self):
        """Start the cleanup task if not already running"""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_old_data())
                logger.info("Started token manager cleanup task")
            except RuntimeError:
                # No event loop running yet, will start later
                logger.info("No event loop available for cleanup task, will start later")
    
    def select_model(self, task_complexity: TaskComplexity, estimated_tokens: int = 1000) -> str:
        """Select the most appropriate model for the task"""
        
        # For large requests (>5000 tokens), use Claude 3-7 Sonnet for better quality
        if estimated_tokens > 5000:
            return ModelType.SONNET_7.value
        
        # For simple tasks, always use Haiku
        if task_complexity == TaskComplexity.SIMPLE:
            return ModelType.HAIKU.value
        
        # For medium tasks, use Sonnet if budget allows
        if task_complexity == TaskComplexity.MEDIUM:
            if self._can_afford_request(ModelType.SONNET.value, estimated_tokens):
                return ModelType.SONNET.value
            else:
                return ModelType.HAIKU.value
        
        # For complex tasks, use Opus if budget allows
        if task_complexity == TaskComplexity.COMPLEX:
            if self._can_afford_request(ModelType.OPUS.value, estimated_tokens):
                return ModelType.OPUS.value
            elif self._can_afford_request(ModelType.SONNET.value, estimated_tokens):
                return ModelType.SONNET.value
            else:
                return ModelType.HAIKU.value
        
        return ModelType.HAIKU.value
    
    def _can_afford_request(self, model: str, estimated_tokens: int) -> bool:
        """Check if we can afford a request within current budget - ALWAYS ALLOW FOR TRACKING"""
        # Always return True - we're only tracking usage, not restricting
        return True
    
    def _reset_budget_if_needed(self, budget: TokenBudget):
        """Reset budget if an hour has passed"""
        if datetime.now() - budget.last_reset > timedelta(hours=1):
            budget.current_usage = 0
            budget.request_count = 0
            budget.last_reset = datetime.now()
            logger.info(f"Reset token budget for model")
    
    def _get_daily_usage(self, model: str) -> int:
        """Get daily token usage for a model"""
        today = datetime.now().date()
        daily_usage = sum(
            usage.total_tokens for usage in self.usage_history
            if usage.model == model and usage.timestamp.date() == today
        )
        return daily_usage
    
    async def reserve_tokens(self, model: str, estimated_tokens: int, request_id: str, session_id: str) -> bool:
        """Reserve tokens for a request - TRACKING ONLY, NO RESTRICTIONS"""
        logger.info(f"Tracking {estimated_tokens} tokens for {model} (request {request_id})")
        
        # Start cleanup task if not already running
        self._start_cleanup_task()
        
        # NO BUDGET OR RATE LIMIT CHECKS - JUST TRACK USAGE
        # Always allow requests, just track the usage
        
        # Ensure model exists in budgets (fallback for unknown models)
        if model not in self.budgets:
            logger.info(f"Adding unknown model {model} to budgets")
            self.budgets[model] = TokenBudget(
                hourly_limit=25000,    # Default limits
                daily_limit=250000,
                max_concurrent_requests=3
            )
            self.rate_limits[model] = RateLimitInfo(
                requests_per_minute=40,
                requests_per_hour=1000,
                current_requests=deque(),
                last_cleanup=datetime.now()
            )
        
        # Track tokens (for monitoring purposes only)
        self.budgets[model].current_usage += estimated_tokens
        self.budgets[model].request_count += 1
        self.active_requests[model] += 1
        
        # Record request start time for tracking
        self.rate_limits[model].current_requests.append(datetime.now())
        
        logger.info(f"Successfully reserved {estimated_tokens} tokens for {model} (request {request_id})")
        return True
    
    def _check_rate_limits(self, model: str) -> bool:
        """Check if request is within rate limits - ALWAYS ALLOW FOR TRACKING"""
        # Always return True - we're only tracking usage, not restricting
        return True
    
    def record_usage(self, model: str, input_tokens: int, output_tokens: int, 
                    request_id: str, task_type: str, session_id: str):
        """Record actual token usage after request completion"""
        total_tokens = input_tokens + output_tokens
        
        usage = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            timestamp=datetime.now(),
            request_id=request_id,
            task_type=task_type,
            session_id=session_id
        )
        
        self.usage_history.append(usage)
        self.session_usage[session_id].append(usage)
        
        # Release reserved tokens
        self.active_requests[model] = max(0, self.active_requests[model] - 1)
        
        logger.info(f"Recorded usage for {model}: {total_tokens} tokens (request {request_id})")
    
    def get_usage_stats(self, session_id: str = None) -> Dict:
        """Get usage statistics"""
        if session_id:
            usage_list = self.session_usage.get(session_id, [])
        else:
            usage_list = self.usage_history
        
        if not usage_list:
            return {"total_tokens": 0, "requests": 0, "models": {}}
        
        total_tokens = sum(usage.total_tokens for usage in usage_list)
        total_requests = len(usage_list)
        
        model_stats = defaultdict(lambda: {"tokens": 0, "requests": 0})
        for usage in usage_list:
            model_stats[usage.model]["tokens"] += usage.total_tokens
            model_stats[usage.model]["requests"] += 1
        
        return {
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "models": dict(model_stats),
            "session_id": session_id
        }
    
    def get_budget_status(self) -> Dict:
        """Get current budget status for all models"""
        status = {}
        for model, budget in self.budgets.items():
            self._reset_budget_if_needed(budget)
            daily_usage = self._get_daily_usage(model)
            
            status[model] = {
                "hourly_usage": budget.current_usage,
                "hourly_limit": budget.hourly_limit,
                "hourly_remaining": budget.hourly_limit - budget.current_usage,
                "daily_usage": daily_usage,
                "daily_limit": budget.daily_limit,
                "daily_remaining": budget.daily_limit - daily_usage,
                "active_requests": self.active_requests[model],
                "max_concurrent": budget.max_concurrent_requests
            }
        
        return status
    
    async def wait_for_budget(self, model: str, estimated_tokens: int, max_wait: int = 300) -> bool:
        """Wait for budget to become available"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self._can_afford_request(model, estimated_tokens):
                return True
            
            # Wait for budget reset or other requests to complete
            await asyncio.sleep(5)
        
        return False
    
    async def _cleanup_old_data(self):
        """Clean up old usage data to prevent memory leaks"""
        while True:
            try:
                # Keep only last 7 days of usage history
                cutoff_date = datetime.now() - timedelta(days=7)
                self.usage_history = [
                    usage for usage in self.usage_history
                    if usage.timestamp > cutoff_date
                ]
                
                # Clean up old session data
                for session_id in list(self.session_usage.keys()):
                    self.session_usage[session_id] = [
                        usage for usage in self.session_usage[session_id]
                        if usage.timestamp > cutoff_date
                    ]
                    
                    # Remove empty sessions
                    if not self.session_usage[session_id]:
                        del self.session_usage[session_id]
                
                logger.info(f"Cleaned up old usage data. Current history: {len(self.usage_history)} records")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
            
            # Run cleanup every hour
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Rough estimation: 1 token ≈ 4 characters for English text
        return max(1, len(text) // 4)
    
    def get_recommended_model(self, task_description: str, estimated_tokens: int = 1000) -> str:
        """Get recommended model based on task description and token estimate"""
        
        # Simple heuristics for task complexity
        if any(keyword in task_description.lower() for keyword in [
            'categorize', 'classify', 'identify', 'prioritize', 'list', 'extract'
        ]):
            complexity = TaskComplexity.SIMPLE
        elif any(keyword in task_description.lower() for keyword in [
            'analyze', 'summarize', 'explain', 'describe', 'review'
        ]):
            complexity = TaskComplexity.MEDIUM
        else:
            complexity = TaskComplexity.COMPLEX
        
        return self.select_model(complexity, estimated_tokens)

# Global token manager instance
token_manager = TokenManager()
