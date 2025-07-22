"""
Chat Session Management for RAG System.

Manages chat sessions, conversation history, and context persistence
for the unified RAG chat interface with DeepSeek-R1:8b integration.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from contextlib import asynccontextmanager

from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class MessageRole(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Individual chat message."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    sources: List[Dict[str, Any]] = None
    confidence: float = None
    processing_time_ms: float = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.sources is None:
            data['sources'] = []
        if self.metadata is None:
            data['metadata'] = {}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['role'] = MessageRole(data['role'])
        return cls(**data)


@dataclass
class ChatSession:
    """Chat session with conversation history and metadata."""
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    conversation_history: List[ChatMessage]
    context: Dict[str, Any]
    settings: Dict[str, Any]
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'conversation_history': [msg.to_dict() for msg in self.conversation_history],
            'context': self.context,
            'settings': self.settings,
            'is_active': self.is_active
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['conversation_history'] = [
            ChatMessage.from_dict(msg) for msg in data['conversation_history']
        ]
        return cls(**data)

    def add_message(self, 
                    role: MessageRole, 
                    content: str, 
                    metadata: Dict[str, Any] = None,
                    sources: List[Dict[str, Any]] = None,
                    confidence: float = None,
                    processing_time_ms: float = None) -> ChatMessage:
        """Add a message to the conversation history."""
        message = ChatMessage(
            id=str(uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            sources=sources or [],
            confidence=confidence,
            processing_time_ms=processing_time_ms
        )
        
        self.conversation_history.append(message)
        self.updated_at = datetime.now()
        return message

    def get_recent_messages(self, limit: int = 20) -> List[ChatMessage]:
        """Get recent messages for context."""
        return self.conversation_history[-limit:] if self.conversation_history else []

    def get_conversation_context(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation context in format suitable for LLM."""
        recent_messages = self.get_recent_messages(limit)
        return [
            {
                "role": msg.role.value,
                "content": msg.content
            }
            for msg in recent_messages
            if msg.role in [MessageRole.USER, MessageRole.ASSISTANT]
        ]


class ChatSessionManager:
    """Manages chat sessions with Redis persistence."""
    
    def __init__(self, redis_url: str = None, session_timeout: int = 3600):
        """
        Initialize chat session manager.
        
        Args:
            redis_url: Redis connection URL
            session_timeout: Session timeout in seconds (default: 1 hour)
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.session_timeout = session_timeout
        self.redis_client: Optional[redis.Redis] = None
        
        # Default session settings
        self.default_settings = {
            "model": "deepseek-r1:8b",
            "temperature": 0.6,
            "max_tokens": 4096,
            "top_p": 0.95,
            "include_dashboard": True,
            "include_documents": True,
            "stream_response": True
        }
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url, 
                encoding="utf-8", 
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Chat session manager initialized with Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    @asynccontextmanager
    async def get_redis(self):
        """Get Redis client with connection handling."""
        if not self.redis_client:
            await self.initialize()
        
        if not self.redis_client:
            raise RuntimeError("Redis client not available")
        
        yield self.redis_client
    
    async def create_session(self, 
                           user_id: str = None, 
                           custom_settings: Dict[str, Any] = None) -> ChatSession:
        """
        Create a new chat session.
        
        Args:
            user_id: Optional user identifier
            custom_settings: Optional custom session settings
            
        Returns:
            New chat session
        """
        session_id = str(uuid4())
        now = datetime.now()
        
        # Merge custom settings with defaults
        settings = self.default_settings.copy()
        if custom_settings:
            settings.update(custom_settings)
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            updated_at=now,
            conversation_history=[],
            context={},
            settings=settings
        )
        
        await self._save_session(session)
        logger.info(f"Created chat session: {session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get chat session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Chat session or None if not found
        """
        try:
            async with self.get_redis() as redis_client:
                data = await redis_client.get(f"chat_session:{session_id}")
                
                if not data:
                    return None
                
                session_data = json.loads(data)
                session = ChatSession.from_dict(session_data)
                return session
                
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    async def save_session(self, session: ChatSession) -> bool:
        """
        Save chat session.
        
        Args:
            session: Chat session to save
            
        Returns:
            True if successful, False otherwise
        """
        return await self._save_session(session)
    
    async def _save_session(self, session: ChatSession) -> bool:
        """Internal method to save session to Redis."""
        try:
            session.updated_at = datetime.now()
            
            async with self.get_redis() as redis_client:
                session_data = json.dumps(session.to_dict(), default=str)
                await redis_client.setex(
                    f"chat_session:{session.session_id}",
                    self.session_timeout,
                    session_data
                )
                
                # Also save to user sessions index if user_id exists
                if session.user_id:
                    await redis_client.sadd(
                        f"user_sessions:{session.user_id}",
                        session.session_id
                    )
                    await redis_client.expire(
                        f"user_sessions:{session.user_id}",
                        self.session_timeout * 2
                    )
                
                return True
                
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {e}")
            return False
    
    async def add_message(self, 
                         session_id: str,
                         role: MessageRole,
                         content: str,
                         metadata: Dict[str, Any] = None,
                         sources: List[Dict[str, Any]] = None,
                         confidence: float = None,
                         processing_time_ms: float = None) -> Optional[ChatMessage]:
        """
        Add message to session conversation history.
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional message metadata
            sources: Optional source information
            confidence: Optional confidence score
            processing_time_ms: Optional processing time
            
        Returns:
            Added message or None if session not found
        """
        session = await self.get_session(session_id)
        if not session:
            return None
        
        message = session.add_message(
            role=role,
            content=content,
            metadata=metadata,
            sources=sources,
            confidence=confidence,
            processing_time_ms=processing_time_ms
        )
        
        # Save updated session
        await self._save_session(session)
        
        logger.info(f"Added {role.value} message to session {session_id}")
        return message
    
    async def get_conversation_history(self, 
                                     session_id: str, 
                                     limit: int = 20) -> List[ChatMessage]:
        """
        Get conversation history for session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages
            
        Returns:
            List of messages
        """
        session = await self.get_session(session_id)
        if not session:
            return []
        
        return session.get_recent_messages(limit)
    
    async def update_session_context(self, 
                                   session_id: str, 
                                   context_update: Dict[str, Any]) -> bool:
        """
        Update session context.
        
        Args:
            session_id: Session identifier
            context_update: Context data to update
            
        Returns:
            True if successful, False otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.context.update(context_update)
        return await self._save_session(session)
    
    async def update_session_settings(self, 
                                    session_id: str, 
                                    settings_update: Dict[str, Any]) -> bool:
        """
        Update session settings.
        
        Args:
            session_id: Session identifier
            settings_update: Settings to update
            
        Returns:
            True if successful, False otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.settings.update(settings_update)
        return await self._save_session(session)
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete chat session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get session first to get user_id for cleanup
            session = await self.get_session(session_id)
            
            async with self.get_redis() as redis_client:
                # Delete main session
                result = await redis_client.delete(f"chat_session:{session_id}")
                
                # Clean up user sessions index
                if session and session.user_id:
                    await redis_client.srem(
                        f"user_sessions:{session.user_id}",
                        session_id
                    )
                
                logger.info(f"Deleted chat session: {session_id}")
                return bool(result)
                
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[str]:
        """
        Get all session IDs for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session IDs
        """
        try:
            async with self.get_redis() as redis_client:
                session_ids = await redis_client.smembers(f"user_sessions:{user_id}")
                return list(session_ids) if session_ids else []
                
        except Exception as e:
            logger.error(f"Error getting user sessions for {user_id}: {e}")
            return []
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            cleanup_count = 0
            
            async with self.get_redis() as redis_client:
                # Get all session keys
                session_keys = await redis_client.keys("chat_session:*")
                
                for key in session_keys:
                    # Redis will automatically expire keys, but we can check TTL
                    ttl = await redis_client.ttl(key)
                    if ttl == -2:  # Key doesn't exist (expired)
                        cleanup_count += 1
                
                logger.info(f"Cleaned up {cleanup_count} expired sessions")
                return cleanup_count
                
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            return 0
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        try:
            async with self.get_redis() as redis_client:
                session_keys = await redis_client.keys("chat_session:*")
                user_keys = await redis_client.keys("user_sessions:*")
                
                # Count active sessions by checking TTL
                active_sessions = 0
                for key in session_keys:
                    ttl = await redis_client.ttl(key)
                    if ttl > 0:
                        active_sessions += 1
                
                return {
                    "total_session_keys": len(session_keys),
                    "active_sessions": active_sessions,
                    "unique_users": len(user_keys),
                    "cleanup_needed": len(session_keys) - active_sessions
                }
                
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {
                "total_session_keys": 0,
                "active_sessions": 0,
                "unique_users": 0,
                "cleanup_needed": 0,
                "error": str(e)
            }


# Global session manager instance
chat_session_manager = ChatSessionManager()


# Startup/shutdown events
async def init_session_manager():
    """Initialize session manager on startup."""
    await chat_session_manager.initialize()


async def cleanup_session_manager():
    """Cleanup session manager on shutdown."""
    await chat_session_manager.close()