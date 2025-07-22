"""
Enhanced document management service for RAG system integration.

This service extends the existing RAG system (qwen_local_rag_agent.py) with:
- Document categorization and metadata tracking
- Enhanced preprocessing pipeline
- Integration with dashboard analytics
- Support for multiple document types
"""

import os
import uuid
import hashlib
import mimetypes
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document

from app.core.logging import get_logger
from app.core.config import get_settings

settings = get_settings()
logger = get_logger(__name__)


class DocumentCategory(str, Enum):
    """Document category classifications."""
    INDUSTRY_REPORT = "industry_report"
    CUSTOMER_FEEDBACK = "customer_feedback"
    MARKET_RESEARCH = "market_research"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    STRATEGY_DOCUMENT = "strategy_document"
    TELCO_ANALYSIS = "telco_analysis"
    CHURN_ANALYSIS = "churn_analysis"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class DocumentMetadata:
    """Enhanced document metadata structure."""
    document_id: str
    filename: str
    original_filename: str
    file_size: int
    file_type: str
    category: DocumentCategory
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = None
    author: Optional[str] = None
    source_url: Optional[str] = None
    upload_timestamp: str = None
    processing_timestamp: Optional[str] = None
    status: DocumentStatus = DocumentStatus.UPLOADED
    chunk_count: int = 0
    embedding_model: str = "snowflake-arctic-embed"
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.upload_timestamp is None:
            self.upload_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Create from dictionary."""
        return cls(**data)


class DocumentPreprocessor:
    """Enhanced document preprocessing pipeline."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def detect_document_category(self, content: str, filename: str) -> DocumentCategory:
        """Intelligently detect document category based on content and filename."""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        # Keywords for different categories
        category_keywords = {
            DocumentCategory.INDUSTRY_REPORT: [
                "industry report", "market analysis", "industry analysis", 
                "annual report", "quarterly report", "sector analysis"
            ],
            DocumentCategory.CUSTOMER_FEEDBACK: [
                "customer feedback", "survey results", "customer satisfaction",
                "nps score", "customer survey", "feedback analysis", "review"
            ],
            DocumentCategory.MARKET_RESEARCH: [
                "market research", "consumer behavior", "market trends",
                "market study", "research findings", "market insights"
            ],
            DocumentCategory.COMPETITOR_ANALYSIS: [
                "competitor analysis", "competitive landscape", "benchmark",
                "competitor report", "market competition", "competitive intelligence"
            ],
            DocumentCategory.TELCO_ANALYSIS: [
                "telecommunications", "telecom", "telco", "mobile network",
                "broadband", "5g", "network infrastructure", "wireless"
            ],
            DocumentCategory.CHURN_ANALYSIS: [
                "churn analysis", "customer churn", "retention", "attrition",
                "churn rate", "customer retention", "churn prediction"
            ]
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in content_lower:
                    score += 2  # Content match gets higher weight
                if keyword in filename_lower:
                    score += 1  # Filename match gets lower weight
            category_scores[category] = score
        
        # Return category with highest score, or OTHER if no good match
        best_category = max(category_scores, key=category_scores.get)
        if category_scores[best_category] > 0:
            return best_category
        return DocumentCategory.OTHER
    
    def extract_metadata_from_content(self, content: str) -> Dict[str, Any]:
        """Extract additional metadata from document content."""
        lines = content.split('\n')
        metadata = {}
        
        # Try to extract title (first non-empty line or line with title indicators)
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and (len(line) > 10 or any(indicator in line.lower() 
                        for indicator in ['title:', 'report:', 'analysis:'])):
                metadata['title'] = line.replace('Title:', '').replace('TITLE:', '').strip()
                break
        
        # Extract potential author information
        author_indicators = ['author:', 'by:', 'prepared by:', 'analyst:']
        for line in lines[:20]:  # Check first 20 lines
            line_lower = line.lower().strip()
            for indicator in author_indicators:
                if indicator in line_lower:
                    author = line.split(':', 1)[-1].strip()
                    if author and len(author) < 100:  # Reasonable author name length
                        metadata['author'] = author
                    break
        
        # Extract date information
        date_indicators = ['date:', 'published:', 'report date:']
        for line in lines[:20]:
            line_lower = line.lower().strip()
            for indicator in date_indicators:
                if indicator in line_lower:
                    date_str = line.split(':', 1)[-1].strip()
                    if date_str and len(date_str) < 50:
                        metadata['document_date'] = date_str
                    break
        
        return metadata
    
    def generate_tags_from_content(self, content: str, category: DocumentCategory) -> List[str]:
        """Generate relevant tags based on content and category."""
        content_lower = content.lower()
        tags = []
        
        # Category-specific tag keywords
        tag_keywords = {
            DocumentCategory.TELCO_ANALYSIS: [
                "5g", "fiber", "broadband", "mobile", "network", "infrastructure",
                "wireless", "spectrum", "towers", "coverage"
            ],
            DocumentCategory.CHURN_ANALYSIS: [
                "retention", "attrition", "satisfaction", "loyalty", "revenue",
                "contract", "pricing", "competition", "service quality"
            ],
            DocumentCategory.CUSTOMER_FEEDBACK: [
                "satisfaction", "nps", "survey", "rating", "complaint",
                "recommendation", "experience", "service", "support"
            ],
            DocumentCategory.MARKET_RESEARCH: [
                "trends", "insights", "demographics", "behavior", "preferences",
                "segmentation", "analysis", "forecast", "growth"
            ],
            DocumentCategory.INDUSTRY_REPORT: [
                "revenue", "growth", "market share", "trends", "forecast",
                "analysis", "performance", "outlook", "strategy"
            ]
        }
        
        # Add category-specific tags
        if category in tag_keywords:
            for keyword in tag_keywords[category]:
                if keyword in content_lower:
                    tags.append(keyword)
        
        # Add general business/telco tags
        general_keywords = {
            "financial": ["revenue", "profit", "cost", "price", "margin"],
            "customer": ["customer", "client", "user", "subscriber"],
            "technology": ["technology", "digital", "innovation", "platform"],
            "service": ["service", "quality", "delivery", "support"],
            "strategy": ["strategy", "plan", "goal", "objective", "target"]
        }
        
        for tag_group, keywords in general_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tag_group)
        
        return list(set(tags))  # Remove duplicates
    
    def process_document(self, file_path: str, metadata: DocumentMetadata) -> Tuple[List[Document], DocumentMetadata]:
        """Process document with enhanced metadata extraction."""
        try:
            # Load document based on file type
            if metadata.file_type == "application/pdf":
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif metadata.file_type.startswith("text/"):
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {metadata.file_type}")
            
            # Combine all document content for analysis
            full_content = "\n".join([doc.page_content for doc in documents])
            
            # Auto-detect category if not specified
            if metadata.category == DocumentCategory.OTHER:
                metadata.category = self.detect_document_category(full_content, metadata.filename)
            
            # Extract metadata from content
            content_metadata = self.extract_metadata_from_content(full_content)
            if not metadata.title and 'title' in content_metadata:
                metadata.title = content_metadata['title']
            if not metadata.author and 'author' in content_metadata:
                metadata.author = content_metadata['author']
            
            # Generate tags
            if not metadata.tags:
                metadata.tags = self.generate_tags_from_content(full_content, metadata.category)
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Enhance each chunk with metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "document_id": metadata.document_id,
                    "chunk_id": f"{metadata.document_id}_chunk_{i}",
                    "category": metadata.category.value,
                    "tags": metadata.tags,
                    "filename": metadata.filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processing_timestamp": datetime.now().isoformat()
                })
            
            # Update metadata
            metadata.chunk_count = len(chunks)
            metadata.status = DocumentStatus.PROCESSED
            metadata.processing_timestamp = datetime.now().isoformat()
            
            logger.info(f"Processed document {metadata.filename}: {len(chunks)} chunks, category: {metadata.category}")
            
            return chunks, metadata
            
        except Exception as e:
            metadata.status = DocumentStatus.FAILED
            logger.error(f"Document processing failed for {metadata.filename}: {e}")
            raise


class EnhancedDocumentManager:
    """Enhanced document management service."""
    
    def __init__(self):
        self.preprocessor = DocumentPreprocessor()
        self.storage_path = Path(settings.MODEL_PATH) / "documents"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        
        # Collection names
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.metadata_collection = f"{self.collection_name}_metadata"
        
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize Qdrant collections for documents and metadata."""
        try:
            # Main document collection (for chunks)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,  # snowflake-arctic-embed dimensions
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.error(f"Failed to create collection {self.collection_name}: {e}")
        
        try:
            # Metadata collection (for document metadata)
            self.qdrant_client.create_collection(
                collection_name=self.metadata_collection,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created metadata collection: {self.metadata_collection}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.error(f"Failed to create metadata collection {self.metadata_collection}: {e}")
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def upload_document(self, 
                       file_path: str, 
                       original_filename: str,
                       category: Optional[DocumentCategory] = None,
                       title: Optional[str] = None,
                       description: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       author: Optional[str] = None) -> DocumentMetadata:
        """Upload and process a document."""
        
        # Generate document ID and new filename
        document_id = str(uuid.uuid4())
        file_extension = Path(original_filename).suffix
        new_filename = f"{document_id}{file_extension}"
        storage_file_path = self.storage_path / new_filename
        
        # Move file to storage location
        import shutil
        shutil.copy2(file_path, storage_file_path)
        
        # Get file information
        file_stats = os.stat(storage_file_path)
        file_type, _ = mimetypes.guess_type(storage_file_path)
        
        # Create metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=new_filename,
            original_filename=original_filename,
            file_size=file_stats.st_size,
            file_type=file_type or "application/octet-stream",
            category=category or DocumentCategory.OTHER,
            title=title,
            description=description,
            tags=tags or [],
            author=author,
            checksum=self._calculate_file_checksum(storage_file_path)
        )
        
        # Process document
        try:
            chunks, updated_metadata = self.preprocessor.process_document(str(storage_file_path), metadata)
            
            # Store chunks in vector database
            self._store_document_chunks(chunks, updated_metadata)
            
            # Store metadata
            self._store_document_metadata(updated_metadata)
            
            logger.info(f"Successfully uploaded and processed document: {original_filename}")
            return updated_metadata
            
        except Exception as e:
            # Clean up on failure
            if storage_file_path.exists():
                storage_file_path.unlink()
            metadata.status = DocumentStatus.FAILED
            logger.error(f"Failed to process document {original_filename}: {e}")
            raise
    
    def _store_document_chunks(self, chunks: List[Document], metadata: DocumentMetadata):
        """Store document chunks in Qdrant vector database."""
        # This will be implemented with the actual embedding model
        # For now, we'll store the metadata and prepare for vector storage
        logger.info(f"Prepared {len(chunks)} chunks for vector storage")
        # TODO: Integrate with existing OllamaEmbedderr from qwen_local_rag_agent.py
    
    def _store_document_metadata(self, metadata: DocumentMetadata):
        """Store document metadata."""
        # Store in a simple JSON file for now (can be replaced with database)
        metadata_file = self.storage_path / f"{metadata.document_id}_metadata.json"
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        logger.info(f"Stored metadata for document: {metadata.document_id}")
    
    def list_documents(self, 
                      category: Optional[DocumentCategory] = None,
                      tags: Optional[List[str]] = None,
                      limit: int = 50) -> List[DocumentMetadata]:
        """List documents with optional filtering."""
        documents = []
        
        # Read all metadata files
        for metadata_file in self.storage_path.glob("*_metadata.json"):
            try:
                import json
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    doc_metadata = DocumentMetadata.from_dict(data)
                    
                    # Apply filters
                    if category and doc_metadata.category != category:
                        continue
                    
                    if tags and not any(tag in doc_metadata.tags for tag in tags):
                        continue
                    
                    documents.append(doc_metadata)
                    
            except Exception as e:
                logger.error(f"Failed to read metadata file {metadata_file}: {e}")
        
        # Sort by upload timestamp (newest first) and limit
        documents.sort(key=lambda x: x.upload_timestamp, reverse=True)
        return documents[:limit]
    
    def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get metadata for a specific document."""
        metadata_file = self.storage_path / f"{document_id}_metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            import json
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return DocumentMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to read metadata for document {document_id}: {e}")
            return None
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its metadata."""
        try:
            metadata = self.get_document_metadata(document_id)
            if not metadata:
                return False
            
            # Delete files
            document_file = self.storage_path / metadata.filename
            metadata_file = self.storage_path / f"{document_id}_metadata.json"
            
            if document_file.exists():
                document_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            
            # TODO: Delete from vector database
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_documents_by_category(self, category: DocumentCategory) -> List[DocumentMetadata]:
        """Get all documents in a specific category."""
        return self.list_documents(category=category)
    
    def search_documents(self, query: str, limit: int = 10) -> List[DocumentMetadata]:
        """Search documents by content and metadata."""
        # Simple implementation - search in titles, descriptions, and tags
        all_documents = self.list_documents(limit=1000)
        query_lower = query.lower()
        
        matches = []
        for doc in all_documents:
            score = 0
            
            # Title match
            if doc.title and query_lower in doc.title.lower():
                score += 3
            
            # Description match
            if doc.description and query_lower in doc.description.lower():
                score += 2
            
            # Tag match
            if any(query_lower in tag.lower() for tag in doc.tags):
                score += 1
            
            # Filename match
            if query_lower in doc.original_filename.lower():
                score += 1
            
            if score > 0:
                matches.append((doc, score))
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in matches[:limit]]


# Global document manager instance
document_manager = EnhancedDocumentManager()