"""
DeepSeek-R1:8b Chat Service for Unified RAG System.

Integrates DeepSeek-R1:8b model via Ollama for streaming chat responses
with the unified RAG system that combines dashboard and document context.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
from datetime import datetime
import httpx
from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.config import get_settings
from app.services.unified_rag_system import unified_rag_system, UnifiedResponse
from app.services.chat_session_manager import ChatSession, MessageRole

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class StreamChunk:
    """Individual streaming response chunk."""
    content: str
    done: bool = False
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "done": self.done,
            "metadata": self.metadata or {},
            "timestamp": datetime.now().isoformat()
        }


class DeepSeekChatService:
    """Service for DeepSeek-R1:8b model integration with RAG."""
    
    def __init__(self):
        """Initialize DeepSeek chat service."""
        self.ollama_base_url = settings.OLLAMA_BASE_URL or "http://localhost:11434"
        self.model_name = "deepseek-r1:8b"
        
        # Default model parameters optimized for DeepSeek-R1
        self.default_parameters = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 4096,
            "repeat_penalty": 1.1,
            "stop": ["</think>", "<|end|>"]
        }
        
        # HTTP client for Ollama API
        self.http_client = None
    
    async def initialize(self):
        """Initialize HTTP client and verify model availability."""
        try:
            self.http_client = httpx.AsyncClient(timeout=60.0)
            
            # Check if model is available
            await self._check_model_availability()
            
            logger.info(f"DeepSeek chat service initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek chat service: {e}")
            if self.http_client:
                await self.http_client.aclose()
                self.http_client = None
            raise
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
    
    async def _check_model_availability(self):
        """Check if DeepSeek-R1:8b model is available in Ollama."""
        try:
            response = await self.http_client.get(f"{self.ollama_base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                raise ValueError(f"Model {self.model_name} not available in Ollama")
            
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise ValueError("Cannot connect to Ollama service")
    
    def _build_deepseek_prompt(self, 
                              user_message: str, 
                              rag_response: UnifiedResponse, 
                              conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Build optimized prompt for DeepSeek-R1:8b with RAG context.
        
        Args:
            user_message: User's question
            rag_response: Unified RAG response with context
            conversation_history: Recent conversation for context
            
        Returns:
            Formatted prompt for DeepSeek
        """
        # Start with thinking section for DeepSeek-R1
        prompt = "<think>\n"
        prompt += f"The user is asking: {user_message}\n\n"
        
        # Add RAG context analysis
        if rag_response.context_pieces:
            prompt += "I have the following context information:\n\n"
            
            # Dashboard analytics context
            dashboard_context = [cp for cp in rag_response.context_pieces 
                               if cp.source.value == "dashboard_analytics"]
            if dashboard_context:
                prompt += "Dashboard Analytics Data:\n"
                for i, context in enumerate(dashboard_context, 1):
                    prompt += f"{i}. {context.content}\n"
                    if context.metadata:
                        prompt += f"   Metadata: {context.metadata}\n"
                prompt += "\n"
            
            # Document context
            document_context = [cp for cp in rag_response.context_pieces 
                              if cp.source.value == "document_content"]
            if document_context:
                prompt += "Document Insights:\n"
                for i, context in enumerate(document_context, 1):
                    prompt += f"{i}. {context.content}\n"
                    if context.source_details:
                        prompt += f"   Source: {context.source_details.get('filename', 'Unknown document')}\n"
                prompt += "\n"
        
        # Add conversation context if available
        if conversation_history:
            prompt += "Recent conversation context:\n"
            for msg in conversation_history[-3:]:  # Last 3 exchanges
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "\n"
        
        # Analysis instructions
        prompt += "I need to:\n"
        prompt += "1. Understand what the user is specifically asking about\n"
        prompt += "2. Identify the most relevant information from the context\n"
        prompt += "3. Synthesize a comprehensive, accurate response\n"
        prompt += "4. Cite sources appropriately\n"
        prompt += "5. Provide actionable insights where applicable\n\n"
        
        # Add query type context
        prompt += f"The query was classified as: {rag_response.query_type.value}\n"
        prompt += f"Sources used: {', '.join(rag_response.sources_used)}\n"
        prompt += f"Confidence level: {rag_response.confidence:.2f}\n"
        
        prompt += "</think>\n\n"
        
        # Main response instruction
        prompt += f"Based on the available context information, I'll provide a comprehensive answer to: {user_message}\n\n"
        
        # Add specific instructions based on query type
        if rag_response.query_type.value == "dashboard_only":
            prompt += "Focus on the current analytics data and metrics provided."
        elif rag_response.query_type.value == "documents_only":
            prompt += "Focus on insights from the uploaded documents and research."
        elif rag_response.query_type.value == "hybrid":
            prompt += "Combine both the current analytics data and document insights to provide a comprehensive view."
        else:
            prompt += "Use all available information to provide the best possible answer."
        
        return prompt
    
    async def generate_streaming_response(self,
                                        user_message: str,
                                        session: ChatSession,
                                        include_dashboard: bool = True,
                                        include_documents: bool = True) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate streaming response using DeepSeek-R1:8b with RAG context.
        
        Args:
            user_message: User's message/question
            session: Chat session with conversation history
            include_dashboard: Whether to include dashboard analytics
            include_documents: Whether to include document context
            
        Yields:
            StreamChunk objects with response content
        """
        if not self.http_client:
            yield StreamChunk(content="Service not initialized", done=True)
            return
        
        start_time = datetime.now()
        
        try:
            # Get RAG context first
            logger.info(f"Getting RAG context for: {user_message}")
            rag_response = await unified_rag_system.query_with_context(
                question=user_message,
                include_dashboard=include_dashboard,
                include_documents=include_documents
            )
            
            # Build conversation history for context
            conversation_context = session.get_conversation_context(limit=6)
            
            # Build optimized prompt for DeepSeek-R1
            deepseek_prompt = self._build_deepseek_prompt(
                user_message=user_message,
                rag_response=rag_response,
                conversation_history=conversation_context
            )
            
            # Prepare model parameters
            model_params = session.settings.copy()
            model_params.update(self.default_parameters)
            
            # Prepare request payload for Ollama
            request_data = {
                "model": self.model_name,
                "prompt": deepseek_prompt,
                "stream": True,
                "options": {
                    "temperature": model_params.get("temperature", 0.6),
                    "top_p": model_params.get("top_p", 0.95),
                    "num_predict": model_params.get("max_tokens", 4096),
                    "repeat_penalty": model_params.get("repeat_penalty", 1.1),
                    "stop": model_params.get("stop", ["</think>", "<|end|>"])
                }
            }
            
            logger.info(f"Starting DeepSeek streaming with params: {model_params}")
            
            # Make streaming request to Ollama
            response_content = ""
            chunk_count = 0
            
            async with self.http_client.stream(
                "POST",
                f"{self.ollama_base_url}/api/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk_data = json.loads(line)
                            
                            if "response" in chunk_data:
                                content = chunk_data["response"]
                                response_content += content
                                chunk_count += 1
                                
                                # Yield chunk with metadata
                                chunk_metadata = {
                                    "chunk_count": chunk_count,
                                    "total_length": len(response_content),
                                    "sources_used": rag_response.sources_used,
                                    "query_type": rag_response.query_type.value,
                                    "rag_confidence": rag_response.confidence
                                }
                                
                                yield StreamChunk(
                                    content=content,
                                    done=chunk_data.get("done", False),
                                    metadata=chunk_metadata
                                )
                                
                                # Stop if done
                                if chunk_data.get("done", False):
                                    break
                        
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse streaming chunk: {e}")
                            continue
            
            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Yield final completion chunk
            final_metadata = {
                "chunk_count": chunk_count,
                "total_length": len(response_content),
                "processing_time_ms": processing_time,
                "sources_used": rag_response.sources_used,
                "query_type": rag_response.query_type.value,
                "rag_confidence": rag_response.confidence,
                "model_used": self.model_name,
                "context_pieces_count": len(rag_response.context_pieces)
            }
            
            yield StreamChunk(
                content="",
                done=True,
                metadata=final_metadata
            )
            
            logger.info(f"DeepSeek streaming completed: {chunk_count} chunks, {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error in DeepSeek streaming: {e}")
            yield StreamChunk(
                content=f"Error generating response: {str(e)}",
                done=True,
                metadata={"error": str(e)}
            )
    
    async def generate_non_streaming_response(self,
                                            user_message: str,
                                            session: ChatSession,
                                            include_dashboard: bool = True,
                                            include_documents: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Generate complete response (non-streaming) for cases where streaming isn't needed.
        
        Args:
            user_message: User's message/question
            session: Chat session with conversation history
            include_dashboard: Whether to include dashboard analytics
            include_documents: Whether to include document context
            
        Returns:
            Tuple of (response_text, metadata)
        """
        start_time = datetime.now()
        
        try:
            # Collect all streaming chunks
            response_parts = []
            final_metadata = {}
            
            async for chunk in self.generate_streaming_response(
                user_message=user_message,
                session=session,
                include_dashboard=include_dashboard,
                include_documents=include_documents
            ):
                if chunk.content:
                    response_parts.append(chunk.content)
                
                if chunk.done and chunk.metadata:
                    final_metadata = chunk.metadata
            
            full_response = "".join(response_parts)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update metadata
            final_metadata.update({
                "non_streaming": True,
                "total_processing_time_ms": processing_time
            })
            
            return full_response, final_metadata
            
        except Exception as e:
            logger.error(f"Error in non-streaming response: {e}")
            return f"Error generating response: {str(e)}", {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health and model availability.
        
        Returns:
            Health status dictionary
        """
        try:
            if not self.http_client:
                return {
                    "status": "unhealthy",
                    "error": "HTTP client not initialized"
                }
            
            # Check Ollama connection
            response = await self.http_client.get(f"{self.ollama_base_url}/api/version")
            response.raise_for_status()
            
            version_info = response.json()
            
            # Check model availability
            models_response = await self.http_client.get(f"{self.ollama_base_url}/api/tags")
            models_response.raise_for_status()
            
            models = models_response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            model_available = self.model_name in available_models
            
            return {
                "status": "healthy" if model_available else "degraded",
                "ollama_version": version_info.get("version", "unknown"),
                "model_name": self.model_name,
                "model_available": model_available,
                "available_models": available_models,
                "base_url": self.ollama_base_url
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.model_name,
                "base_url": self.ollama_base_url
            }


# Global service instance
deepseek_chat_service = DeepSeekChatService()


# Startup/shutdown events
async def init_deepseek_service():
    """Initialize DeepSeek service on startup."""
    await deepseek_chat_service.initialize()


async def cleanup_deepseek_service():
    """Cleanup DeepSeek service on shutdown."""
    await deepseek_chat_service.close()