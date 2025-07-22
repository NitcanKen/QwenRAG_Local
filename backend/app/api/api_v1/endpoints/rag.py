"""
RAG (Retrieval-Augmented Generation) endpoints with enhanced document management.
"""

import os
import tempfile
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from app.core.logging import get_logger
from app.services.document_management import (
    document_manager, DocumentCategory, DocumentMetadata, DocumentStatus
)
from app.services.chat_session_manager import (
    chat_session_manager, MessageRole, ChatSession
)
from app.services.deepseek_chat_service import deepseek_chat_service

logger = get_logger(__name__)
router = APIRouter()


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    document_id: str
    filename: str
    category: str
    chunk_count: int
    status: str
    message: str


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    success: bool
    documents: List[Dict[str, Any]]
    total_count: int
    filters_applied: Dict[str, Any]


class DocumentSearchRequest(BaseModel):
    """Request model for document search."""
    query: str
    limit: int = 10


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # Comma-separated tags
    author: Optional[str] = Form(None)
) -> DocumentUploadResponse:
    """
    Upload a document for RAG processing with enhanced metadata.
    
    Args:
        file: Document file (PDF, TXT)
        category: Document category (industry_report, customer_feedback, etc.)
        title: Document title
        description: Document description
        tags: Comma-separated tags
        author: Document author
        
    Returns:
        Upload response with document metadata
    """
    try:
        # Validate file type
        if not file.content_type:
            raise HTTPException(status_code=400, detail="Unable to determine file type")
        
        supported_types = ["application/pdf", "text/plain", "text/csv"]
        if file.content_type not in supported_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Supported: {supported_types}"
            )
        
        # Parse category
        doc_category = DocumentCategory.OTHER
        if category:
            try:
                doc_category = DocumentCategory(category.lower())
            except ValueError:
                logger.warning(f"Invalid category '{category}', using 'other'")
        
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            metadata = document_manager.upload_document(
                file_path=tmp_file_path,
                original_filename=file.filename,
                category=doc_category,
                title=title,
                description=description,
                tags=tag_list,
                author=author
            )
            
            logger.info(f"Successfully uploaded document: {file.filename} -> {metadata.document_id}")
            
            return DocumentUploadResponse(
                success=True,
                document_id=metadata.document_id,
                filename=metadata.original_filename,
                category=metadata.category.value,
                chunk_count=metadata.chunk_count,
                status=metadata.status.value,
                message=f"Document '{file.filename}' uploaded and processed successfully"
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    category: Optional[str] = Query(None, description="Filter by category"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of documents to return")
) -> DocumentListResponse:
    """
    List uploaded documents with optional filtering.
    
    Args:
        category: Filter by document category
        tags: Filter by tags (comma-separated)
        limit: Maximum number of documents to return
        
    Returns:
        List of documents with metadata
    """
    try:
        # Parse filters
        filter_category = None
        if category:
            try:
                filter_category = DocumentCategory(category.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        filter_tags = None
        if tags:
            filter_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Get documents
        documents = document_manager.list_documents(
            category=filter_category,
            tags=filter_tags,
            limit=limit
        )
        
        # Convert to dict format
        document_dicts = [doc.to_dict() for doc in documents]
        
        logger.info(f"Listed {len(documents)} documents with filters: category={category}, tags={tags}")
        
        return DocumentListResponse(
            success=True,
            documents=document_dicts,
            total_count=len(documents),
            filters_applied={
                "category": category,
                "tags": tags,
                "limit": limit
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(document_id: str) -> Dict[str, Any]:
    """
    Get metadata for a specific document.
    
    Args:
        document_id: Document identifier
        
    Returns:
        Document metadata
    """
    try:
        metadata = document_manager.get_document_metadata(document_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return JSONResponse(
            content={
                "success": True,
                "document": metadata.to_dict()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str) -> Dict[str, Any]:
    """
    Delete a document and its metadata.
    
    Args:
        document_id: Document identifier
        
    Returns:
        Deletion status
    """
    try:
        success = document_manager.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Successfully deleted document: {document_id}")
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"Document {document_id} deleted successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/documents/search")
async def search_documents(request: DocumentSearchRequest) -> Dict[str, Any]:
    """
    Search documents by content and metadata.
    
    Args:
        request: Search request with query and limit
        
    Returns:
        Search results
    """
    try:
        documents = document_manager.search_documents(
            query=request.query,
            limit=request.limit
        )
        
        # Convert to dict format
        document_dicts = [doc.to_dict() for doc in documents]
        
        logger.info(f"Search for '{request.query}' returned {len(documents)} results")
        
        return JSONResponse(
            content={
                "success": True,
                "query": request.query,
                "results": document_dicts,
                "total_found": len(documents)
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search documents: {str(e)}")


@router.get("/documents/categories")
async def get_document_categories() -> Dict[str, Any]:
    """
    Get available document categories.
    
    Returns:
        List of available categories
    """
    try:
        categories = [
            {
                "value": category.value,
                "label": category.value.replace("_", " ").title(),
                "description": _get_category_description(category)
            }
            for category in DocumentCategory
        ]
        
        return JSONResponse(
            content={
                "success": True,
                "categories": categories
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get categories")


@router.get("/documents/stats")
async def get_document_stats() -> Dict[str, Any]:
    """
    Get document statistics.
    
    Returns:
        Document statistics by category and status
    """
    try:
        all_documents = document_manager.list_documents(limit=10000)
        
        # Calculate statistics
        stats = {
            "total_documents": len(all_documents),
            "by_category": {},
            "by_status": {},
            "recent_uploads": 0
        }
        
        # Count by category and status
        for doc in all_documents:
            # Category stats
            category = doc.category.value
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            # Status stats
            status = doc.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            # Recent uploads (last 7 days)
            from datetime import datetime, timedelta
            upload_date = datetime.fromisoformat(doc.upload_timestamp.replace('Z', '+00:00'))
            if upload_date > datetime.now() - timedelta(days=7):
                stats["recent_uploads"] += 1
        
        return JSONResponse(
            content={
                "success": True,
                "statistics": stats
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document statistics")


def _get_category_description(category: DocumentCategory) -> str:
    """Get description for document category."""
    descriptions = {
        DocumentCategory.INDUSTRY_REPORT: "Industry analysis and market reports",
        DocumentCategory.CUSTOMER_FEEDBACK: "Customer surveys, reviews, and feedback",
        DocumentCategory.MARKET_RESEARCH: "Market research and consumer behavior studies",
        DocumentCategory.COMPETITOR_ANALYSIS: "Competitive intelligence and benchmarking",
        DocumentCategory.STRATEGY_DOCUMENT: "Internal strategy and planning documents",
        DocumentCategory.TELCO_ANALYSIS: "Telecommunications industry analysis",
        DocumentCategory.CHURN_ANALYSIS: "Customer churn and retention analysis",
        DocumentCategory.OTHER: "Other document types"
    }
    return descriptions.get(category, "No description available")


# Legacy endpoints for backwards compatibility
@router.post("/upload")
async def upload_document_legacy(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Legacy upload endpoint for backwards compatibility.
    
    Args:
        file: Uploaded document file
        
    Returns:
        Upload status
    """
    try:
        # Use the new upload endpoint with default parameters
        response = await upload_document(file=file)
        
        # Convert to legacy format
        return JSONResponse(
            content={
                "message": f"Document '{file.filename}' uploaded successfully",
                "file_info": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": file.size if hasattr(file, 'size') else None
                },
                "document_id": response.document_id,
                "status": response.status,
                "chunks_created": response.chunk_count
            }
        )
        
    except Exception as e:
        logger.error(f"Error in legacy upload: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "Upload failed",
                "error": str(e)
            }
        )


@router.post("/chat")
async def chat_with_rag(
    message: str,
    include_dashboard: bool = True,
    include_documents: bool = True
) -> Dict[str, Any]:
    """
    Chat with the unified RAG system using dashboard data and documents.
    
    This endpoint combines dashboard analytics with document insights to provide
    comprehensive, context-aware responses.
    
    Args:
        message: User's question/message
        include_dashboard: Whether to include dashboard data in context
        include_documents: Whether to include uploaded documents in context
        
    Returns:
        Dictionary containing unified RAG response with sources
    """
    try:
        from app.services.unified_rag_system import unified_rag_system
        
        logger.info(f"Unified RAG chat request: {message}")
        
        # Process query through unified RAG system
        response = await unified_rag_system.query_with_context(
            question=message,
            include_dashboard=include_dashboard,
            include_documents=include_documents
        )
        
        # Format sources for response
        sources = []
        for context_piece in response.context_pieces:
            source_info = {
                "type": context_piece.source.value,
                "content_preview": context_piece.content[:200] + "..." if len(context_piece.content) > 200 else context_piece.content,
                "relevance_score": context_piece.relevance_score,
                "metadata": context_piece.metadata
            }
            
            # Add source-specific details
            if context_piece.source_details:
                if context_piece.source.value == "dashboard_analytics":
                    source_info["endpoint"] = context_piece.source_details.get("endpoint")
                elif context_piece.source.value == "document_content":
                    source_info["document_id"] = context_piece.source_details.get("document_id")
                    source_info["filename"] = context_piece.source_details.get("filename")
                    source_info["category"] = context_piece.source_details.get("category")
            
            sources.append(source_info)
        
        logger.info(f"Unified RAG response generated: {response.query_type}, confidence: {response.confidence:.2f}")
        
        return JSONResponse(
            content={
                "success": True,
                "query": message,
                "query_id": response.query_id,
                "query_type": response.query_type.value,
                "settings": {
                    "include_dashboard": include_dashboard,
                    "include_documents": include_documents
                },
                "response": response.answer,
                "sources": sources,
                "sources_used": response.sources_used,
                "confidence": response.confidence,
                "processing_time_ms": response.processing_time_ms,
                "context_pieces_count": len(response.context_pieces),
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error in unified RAG chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {str(e)}")


@router.post("/chat/analyze")
async def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analyze a query to understand how it would be processed.
    
    This endpoint helps users understand how their query will be routed
    and what sources will be used without actually processing it.
    
    Args:
        query: Query to analyze
        
    Returns:
        Analysis of query routing and expected sources
    """
    try:
        from app.services.unified_rag_system import QueryClassifier
        
        classifier = QueryClassifier()
        query_type = classifier.classify_query(query)
        
        # Provide insights about how the query would be processed
        analysis = {
            "query": query,
            "detected_type": query_type.value,
            "routing_strategy": _get_routing_description(query_type),
            "expected_sources": _get_expected_sources(query_type),
            "optimization_tips": _get_optimization_tips(query, query_type)
        }
        
        return JSONResponse(
            content={
                "success": True,
                "analysis": analysis
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze query: {str(e)}")


def _get_routing_description(query_type) -> str:
    """Get description of routing strategy for query type."""
    descriptions = {
        "dashboard_only": "This query will primarily use dashboard analytics data including churn rates, customer segments, and real-time metrics.",
        "documents_only": "This query will search through uploaded documents including reports, research, and feedback.",
        "hybrid": "This query will combine both dashboard analytics and document insights for a comprehensive response.",
        "unknown": "This query type is unclear and will attempt to use all available sources."
    }
    return descriptions.get(query_type.value, "Unknown routing strategy.")


def _get_expected_sources(query_type) -> List[str]:
    """Get expected sources for query type."""
    source_mapping = {
        "dashboard_only": ["dashboard_analytics"],
        "documents_only": ["document_content"],
        "hybrid": ["dashboard_analytics", "document_content"],
        "unknown": ["dashboard_analytics", "document_content"]
    }
    return source_mapping.get(query_type.value, [])


# Chat Interface Models
class ChatSessionRequest(BaseModel):
    """Request model for creating a new chat session."""
    user_id: Optional[str] = None
    custom_settings: Optional[Dict[str, Any]] = None


class ChatSessionResponse(BaseModel):
    """Response model for chat session operations."""
    success: bool
    session_id: str
    message: str
    settings: Dict[str, Any] = {}


class ChatMessageRequest(BaseModel):
    """Request model for chat messages."""
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    include_dashboard: bool = True
    include_documents: bool = True
    stream_response: bool = True


class ChatMessageResponse(BaseModel):
    """Response model for chat messages (non-streaming)."""
    success: bool
    session_id: str
    message_id: str
    response: str
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    timestamp: str


class WebSocketConnectionManager:
    """Manages WebSocket connections for real-time chat."""
    
    def __init__(self):
        # Active connections: {session_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection and store it."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        """Send message to specific session."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending WebSocket message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def broadcast_to_session(self, session_id: str, message_type: str, data: Any):
        """Broadcast message to session with specific type."""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_message(session_id, message)

# Global WebSocket manager
websocket_manager = WebSocketConnectionManager()


# Chat Session Endpoints
@router.post("/chat/session", response_model=ChatSessionResponse)
async def create_chat_session(request: ChatSessionRequest) -> ChatSessionResponse:
    """
    Create a new chat session with DeepSeek-R1:8b integration.
    
    Args:
        request: Session creation request
        
    Returns:
        New session details
    """
    try:
        session = await chat_session_manager.create_session(
            user_id=request.user_id,
            custom_settings=request.custom_settings
        )
        
        logger.info(f"Created chat session: {session.session_id} for user: {request.user_id}")
        
        return ChatSessionResponse(
            success=True,
            session_id=session.session_id,
            message="Chat session created successfully",
            settings=session.settings
        )
        
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create chat session: {str(e)}")


@router.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str) -> Dict[str, Any]:
    """
    Get chat session details and conversation history.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session details with conversation history
    """
    try:
        session = await chat_session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        return JSONResponse(
            content={
                "success": True,
                "session": {
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "settings": session.settings,
                    "message_count": len(session.conversation_history),
                    "is_active": session.is_active
                },
                "conversation_history": [msg.to_dict() for msg in session.conversation_history],
                "context": session.context
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str) -> Dict[str, Any]:
    """
    Delete a chat session and its conversation history.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        success = await chat_session_manager.delete_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Disconnect any active WebSocket
        websocket_manager.disconnect(session_id)
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"Chat session {session_id} deleted successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


# Streaming Chat Endpoints
@router.post("/chat/stream")
async def stream_chat_response(request: ChatMessageRequest):
    """
    Stream chat response using DeepSeek-R1:8b with RAG context.
    
    This endpoint provides Server-Sent Events (SSE) streaming for real-time
    chat responses that combine dashboard analytics and document insights.
    
    Args:
        request: Chat message request
        
    Returns:
        Streaming response with chat content
    """
    try:
        # Get or create session
        if request.session_id:
            session = await chat_session_manager.get_session(request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            session = await chat_session_manager.create_session(user_id=request.user_id)
        
        session_id = session.session_id
        
        # Add user message to session
        user_message = await chat_session_manager.add_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message
        )
        
        logger.info(f"Starting streaming chat for session {session_id}: {request.message}")
        
        async def generate_stream():
            """Generate SSE stream for chat response."""
            try:
                # Send initial acknowledgment
                yield f"event: chat_start\ndata: {json.dumps({'session_id': session_id, 'message_id': user_message.id})}\n\n"
                
                # Generate streaming response
                response_content = ""
                final_metadata = {}
                
                async for chunk in deepseek_chat_service.generate_streaming_response(
                    user_message=request.message,
                    session=session,
                    include_dashboard=request.include_dashboard,
                    include_documents=request.include_documents
                ):
                    if chunk.content:
                        response_content += chunk.content
                        
                        # Send content chunk
                        event_data = {
                            "content": chunk.content,
                            "session_id": session_id,
                            "metadata": chunk.metadata or {}
                        }
                        yield f"event: chat_chunk\ndata: {json.dumps(event_data)}\n\n"
                    
                    if chunk.done:
                        final_metadata = chunk.metadata or {}
                        break
                
                # Save assistant response to session
                assistant_message = await chat_session_manager.add_message(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=response_content,
                    metadata=final_metadata,
                    sources=final_metadata.get("sources_used", []),
                    confidence=final_metadata.get("rag_confidence"),
                    processing_time_ms=final_metadata.get("processing_time_ms")
                )
                
                # Send completion event
                completion_data = {
                    "session_id": session_id,
                    "message_id": assistant_message.id,
                    "total_length": len(response_content),
                    "metadata": final_metadata
                }
                yield f"event: chat_complete\ndata: {json.dumps(completion_data)}\n\n"
                
                logger.info(f"Completed streaming chat for session {session_id}")
                
            except Exception as e:
                logger.error(f"Error in streaming chat: {e}")
                error_data = {
                    "session_id": session_id,
                    "error": str(e)
                }
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        
        return EventSourceResponse(generate_stream())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stream chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stream chat: {str(e)}")


@router.post("/chat", response_model=ChatMessageResponse)
async def chat_with_deepseek(request: ChatMessageRequest) -> ChatMessageResponse:
    """
    Send a chat message and get a complete response (non-streaming).
    
    This endpoint provides a traditional request-response pattern for cases
    where streaming is not needed or supported.
    
    Args:
        request: Chat message request
        
    Returns:
        Complete chat response
    """
    try:
        # Get or create session
        if request.session_id:
            session = await chat_session_manager.get_session(request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            session = await chat_session_manager.create_session(user_id=request.user_id)
        
        session_id = session.session_id
        
        # Add user message to session
        user_message = await chat_session_manager.add_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message
        )
        
        logger.info(f"Processing chat message for session {session_id}: {request.message}")
        
        # Generate response
        response_text, metadata = await deepseek_chat_service.generate_non_streaming_response(
            user_message=request.message,
            session=session,
            include_dashboard=request.include_dashboard,
            include_documents=request.include_documents
        )
        
        # Save assistant response
        assistant_message = await chat_session_manager.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            metadata=metadata,
            sources=metadata.get("sources_used", []),
            confidence=metadata.get("rag_confidence"),
            processing_time_ms=metadata.get("total_processing_time_ms")
        )
        
        logger.info(f"Completed chat message for session {session_id}")
        
        return ChatMessageResponse(
            success=True,
            session_id=session_id,
            message_id=assistant_message.id,
            response=response_text,
            sources=metadata.get("sources_used", []),
            metadata=metadata,
            timestamp=assistant_message.timestamp.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")


# WebSocket Chat Endpoint
@router.websocket("/chat/ws/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat with DeepSeek-R1:8b.
    
    Provides bidirectional real-time communication for chat sessions.
    
    Args:
        websocket: WebSocket connection
        session_id: Chat session identifier
    """
    await websocket_manager.connect(websocket, session_id)
    
    try:
        # Get session
        session = await chat_session_manager.get_session(session_id)
        if not session:
            await websocket_manager.send_message(session_id, {
                "type": "error",
                "message": "Session not found"
            })
            return
        
        # Send connection confirmation
        await websocket_manager.broadcast_to_session(session_id, "connected", {
            "session_id": session_id,
            "message": "WebSocket connected successfully"
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message_type = message_data.get("type")
            
            if message_type == "chat_message":
                # Process chat message
                user_message_content = message_data.get("content", "")
                
                # Add user message to session
                await chat_session_manager.add_message(
                    session_id=session_id,
                    role=MessageRole.USER,
                    content=user_message_content
                )
                
                # Send acknowledgment
                await websocket_manager.broadcast_to_session(session_id, "message_received", {
                    "content": user_message_content
                })
                
                # Generate and stream response
                response_content = ""
                async for chunk in deepseek_chat_service.generate_streaming_response(
                    user_message=user_message_content,
                    session=session,
                    include_dashboard=message_data.get("include_dashboard", True),
                    include_documents=message_data.get("include_documents", True)
                ):
                    if chunk.content:
                        response_content += chunk.content
                        await websocket_manager.broadcast_to_session(session_id, "stream_chunk", {
                            "content": chunk.content,
                            "metadata": chunk.metadata
                        })
                    
                    if chunk.done:
                        # Save assistant response
                        await chat_session_manager.add_message(
                            session_id=session_id,
                            role=MessageRole.ASSISTANT,
                            content=response_content,
                            metadata=chunk.metadata,
                            confidence=chunk.metadata.get("rag_confidence") if chunk.metadata else None,
                            processing_time_ms=chunk.metadata.get("processing_time_ms") if chunk.metadata else None
                        )
                        
                        await websocket_manager.broadcast_to_session(session_id, "stream_complete", {
                            "total_length": len(response_content),
                            "metadata": chunk.metadata
                        })
                        break
            
            elif message_type == "heartbeat":
                # Respond to heartbeat
                await websocket_manager.broadcast_to_session(session_id, "heartbeat_ack", {})
            
            elif message_type == "update_settings":
                # Update session settings
                new_settings = message_data.get("settings", {})
                success = await chat_session_manager.update_session_settings(session_id, new_settings)
                await websocket_manager.broadcast_to_session(session_id, "settings_updated", {
                    "success": success,
                    "settings": new_settings
                })
            
            else:
                # Unknown message type
                await websocket_manager.broadcast_to_session(session_id, "error", {
                    "message": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected for session: {session_id}")
    
    except Exception as e:
        logger.error(f"Error in WebSocket for session {session_id}: {e}")
        await websocket_manager.broadcast_to_session(session_id, "error", {
            "message": str(e)
        })
        websocket_manager.disconnect(session_id)


# Chat Management Endpoints
@router.get("/chat/sessions")
async def list_user_sessions(user_id: str) -> Dict[str, Any]:
    """
    List all chat sessions for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        List of user's chat sessions
    """
    try:
        session_ids = await chat_session_manager.get_user_sessions(user_id)
        
        # Get session details
        sessions = []
        for session_id in session_ids:
            session = await chat_session_manager.get_session(session_id)
            if session:
                sessions.append({
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "message_count": len(session.conversation_history),
                    "is_active": session.is_active
                })
        
        return JSONResponse(
            content={
                "success": True,
                "user_id": user_id,
                "sessions": sessions,
                "total_count": len(sessions)
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing sessions for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get("/chat/health")
async def chat_service_health() -> Dict[str, Any]:
    """
    Check chat service health status.
    
    Returns:
        Health status of all chat components
    """
    try:
        # Check DeepSeek service
        deepseek_health = await deepseek_chat_service.health_check()
        
        # Check session manager
        session_stats = await chat_session_manager.get_session_stats()
        
        # Check WebSocket manager
        websocket_stats = {
            "active_connections": len(websocket_manager.active_connections),
            "connected_sessions": list(websocket_manager.active_connections.keys())
        }
        
        overall_status = "healthy"
        if deepseek_health.get("status") != "healthy":
            overall_status = "degraded"
        
        return JSONResponse(
            content={
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "deepseek_service": deepseek_health,
                    "session_manager": session_stats,
                    "websocket_manager": websocket_stats
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error checking chat health: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


def _get_optimization_tips(query: str, query_type) -> List[str]:
    """Get optimization tips for the query."""
    tips = []
    
    if query_type.value == "unknown":
        tips.append("Try being more specific about whether you want current data or research insights.")
        tips.append("Use keywords like 'analytics', 'dashboard' for current data or 'research', 'document' for insights.")
    
    if "why" in query.lower() or "explain" in query.lower():
        tips.append("Consider asking follow-up questions to drill down into specific aspects.")
    
    if len(query.split()) < 5:
        tips.append("Longer, more descriptive questions often yield better results.")
    
    query_lower = query.lower()
    if not any(keyword in query_lower for keyword in ['churn', 'customer', 'retention', 'analytics']):
        tips.append("Include telco-specific terms like 'churn', 'retention', or 'customer' for better results.")
    
    return tips if tips else ["Your query looks well-formed for optimal processing."]