"""
Unified RAG System for Dashboard-Document Integration.

This system combines dashboard analytics data with uploaded documents
to provide comprehensive, context-aware responses that can answer
questions using both numerical data and document insights.

Example queries:
- "Why are fiber optic customers churning according to market research?"
- "What do our churn analytics show compared to industry reports?"
- "How do customer feedback documents explain our retention rates?"
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from app.core.logging import get_logger
from app.services.analytics import ChurnAnalyticsService
from app.services.document_management import document_manager, DocumentCategory
from app.services.ml_pipeline import ml_service

logger = get_logger(__name__)


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    DASHBOARD_ONLY = "dashboard_only"
    DOCUMENTS_ONLY = "documents_only"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class ContextSource(str, Enum):
    """Sources of context information."""
    DASHBOARD_ANALYTICS = "dashboard_analytics"
    DOCUMENT_CONTENT = "document_content"
    ML_PREDICTIONS = "ml_predictions"
    COMBINED = "combined"


@dataclass
class ContextPiece:
    """A piece of context from a specific source."""
    source: ContextSource
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source_details: Dict[str, Any]


@dataclass
class UnifiedResponse:
    """Unified response combining multiple context sources."""
    answer: str
    query_type: QueryType
    context_pieces: List[ContextPiece]
    confidence: float
    sources_used: List[str]
    processing_time_ms: float
    query_id: str


class QueryClassifier:
    """Classifies queries to determine optimal routing strategy."""
    
    def __init__(self):
        # Keywords that indicate dashboard/analytics queries
        self.dashboard_keywords = [
            'churn rate', 'analytics', 'metrics', 'statistics', 'numbers',
            'percentage', 'trend', 'data', 'analysis', 'breakdown',
            'demographic', 'segment', 'financial', 'revenue', 'tenure',
            'contract', 'payment method', 'monthly charges', 'total charges',
            'customers by', 'show me', 'how many', 'what percentage'
        ]
        
        # Keywords that indicate document queries
        self.document_keywords = [
            'report', 'research', 'study', 'document', 'analysis',
            'feedback', 'survey', 'industry', 'market', 'competitor',
            'according to', 'mentioned in', 'document says', 'research shows',
            'feedback indicates', 'study suggests', 'report states'
        ]
        
        # Keywords that indicate hybrid queries
        self.hybrid_keywords = [
            'compare', 'versus', 'against', 'compared to', 'why',
            'explain', 'reason', 'cause', 'insight', 'perspective',
            'what does', 'how do', 'validate', 'support', 'contradict'
        ]
    
    def classify_query(self, query: str) -> QueryType:
        """Classify query to determine routing strategy."""
        query_lower = query.lower()
        
        dashboard_score = sum(1 for keyword in self.dashboard_keywords 
                             if keyword in query_lower)
        document_score = sum(1 for keyword in self.document_keywords 
                            if keyword in query_lower)
        hybrid_score = sum(1 for keyword in self.hybrid_keywords 
                          if keyword in query_lower)
        
        # Determine query type based on scores
        if hybrid_score > 0 and (dashboard_score > 0 or document_score > 0):
            return QueryType.HYBRID
        elif dashboard_score > document_score:
            return QueryType.DASHBOARD_ONLY
        elif document_score > dashboard_score:
            return QueryType.DOCUMENTS_ONLY
        elif hybrid_score > 0:
            return QueryType.HYBRID
        else:
            return QueryType.UNKNOWN


class DashboardContextProvider:
    """Provides context from dashboard analytics data."""
    
    def __init__(self):
        self.analytics_service = ChurnAnalyticsService()
    
    async def get_context(self, query: str) -> List[ContextPiece]:
        """Get relevant dashboard context for the query."""
        context_pieces = []
        
        try:
            # Determine what analytics to include based on query
            query_lower = query.lower()
            
            # Churn overview if asking about general churn
            if any(keyword in query_lower for keyword in ['churn', 'overall', 'general', 'total']):
                churn_overview = await self.analytics_service.get_churn_overview()
                context_pieces.append(ContextPiece(
                    source=ContextSource.DASHBOARD_ANALYTICS,
                    content=f"Overall churn rate: {churn_overview.get('churn_rate', 0):.1%}. "
                           f"Total customers: {churn_overview.get('total_customers', 0):,}. "
                           f"Churned customers: {churn_overview.get('churned_customers', 0):,}.",
                    metadata={'type': 'churn_overview'},
                    relevance_score=0.9,
                    source_details={'endpoint': '/analytics/churn-rate', 'data': churn_overview}
                ))
            
            # Tenure analysis
            if any(keyword in query_lower for keyword in ['tenure', 'time', 'length', 'duration']):
                tenure_data = await self.analytics_service.get_churn_by_tenure()
                tenure_insights = self._format_tenure_data(tenure_data)
                context_pieces.append(ContextPiece(
                    source=ContextSource.DASHBOARD_ANALYTICS,
                    content=tenure_insights,
                    metadata={'type': 'tenure_analysis'},
                    relevance_score=0.8,
                    source_details={'endpoint': '/analytics/demographics', 'data': tenure_data}
                ))
            
            # Contract analysis
            if any(keyword in query_lower for keyword in ['contract', 'month-to-month', 'yearly', 'commitment']):
                contract_data = await self.analytics_service.get_churn_by_contract()
                contract_insights = self._format_contract_data(contract_data)
                context_pieces.append(ContextPiece(
                    source=ContextSource.DASHBOARD_ANALYTICS,
                    content=contract_insights,
                    metadata={'type': 'contract_analysis'},
                    relevance_score=0.8,
                    source_details={'endpoint': '/analytics/services', 'data': contract_data}
                ))
            
            # Service-specific analysis
            if any(keyword in query_lower for keyword in ['fiber', 'dsl', 'internet', 'service', 'broadband']):
                service_data = await self.analytics_service.get_churn_by_internet_service()
                service_insights = self._format_service_data(service_data)
                context_pieces.append(ContextPiece(
                    source=ContextSource.DASHBOARD_ANALYTICS,
                    content=service_insights,
                    metadata={'type': 'service_analysis'},
                    relevance_score=0.9,
                    source_details={'endpoint': '/analytics/services', 'data': service_data}
                ))
            
            # Demographics
            if any(keyword in query_lower for keyword in ['senior', 'age', 'gender', 'demographic', 'partner', 'family']):
                demo_data = await self.analytics_service.get_demographic_analysis()
                demo_insights = self._format_demographic_data(demo_data)
                context_pieces.append(ContextPiece(
                    source=ContextSource.DASHBOARD_ANALYTICS,
                    content=demo_insights,
                    metadata={'type': 'demographic_analysis'},
                    relevance_score=0.7,
                    source_details={'endpoint': '/analytics/demographics', 'data': demo_data}
                ))
            
            # If no specific matches, get general overview
            if not context_pieces:
                churn_overview = await self.analytics_service.get_churn_overview()
                context_pieces.append(ContextPiece(
                    source=ContextSource.DASHBOARD_ANALYTICS,
                    content=f"Current dashboard shows {churn_overview.get('churn_rate', 0):.1%} churn rate "
                           f"across {churn_overview.get('total_customers', 0):,} customers.",
                    metadata={'type': 'general_overview'},
                    relevance_score=0.6,
                    source_details={'endpoint': '/analytics/churn-rate', 'data': churn_overview}
                ))
            
        except Exception as e:
            logger.error(f"Error getting dashboard context: {e}")
            # Provide fallback context
            context_pieces.append(ContextPiece(
                source=ContextSource.DASHBOARD_ANALYTICS,
                content="Dashboard analytics are currently unavailable.",
                metadata={'type': 'error', 'error': str(e)},
                relevance_score=0.1,
                source_details={'error': str(e)}
            ))
        
        return context_pieces
    
    def _format_tenure_data(self, data: Dict) -> str:
        """Format tenure analysis data."""
        if not data:
            return "Tenure data unavailable."
        
        insights = []
        for tenure_group, stats in data.items():
            if isinstance(stats, dict) and 'churn_rate' in stats:
                insights.append(f"{tenure_group}: {stats['churn_rate']:.1%} churn rate")
        
        return f"Churn by tenure: {', '.join(insights)}."
    
    def _format_contract_data(self, data: Dict) -> str:
        """Format contract analysis data."""
        if not data:
            return "Contract data unavailable."
        
        insights = []
        for contract_type, stats in data.items():
            if isinstance(stats, dict) and 'churn_rate' in stats:
                insights.append(f"{contract_type}: {stats['churn_rate']:.1%}")
        
        return f"Churn by contract type: {', '.join(insights)}."
    
    def _format_service_data(self, data: Dict) -> str:
        """Format service analysis data."""
        if not data:
            return "Service data unavailable."
        
        insights = []
        for service_type, stats in data.items():
            if isinstance(stats, dict) and 'churn_rate' in stats:
                insights.append(f"{service_type}: {stats['churn_rate']:.1%}")
        
        return f"Churn by internet service: {', '.join(insights)}."
    
    def _format_demographic_data(self, data: Dict) -> str:
        """Format demographic analysis data."""
        if not data:
            return "Demographic data unavailable."
        
        insights = []
        if 'gender_analysis' in data:
            for gender, stats in data['gender_analysis'].items():
                if isinstance(stats, dict) and 'churn_rate' in stats:
                    insights.append(f"{gender}: {stats['churn_rate']:.1%}")
        
        if 'senior_citizen_analysis' in data:
            for category, stats in data['senior_citizen_analysis'].items():
                if isinstance(stats, dict) and 'churn_rate' in stats:
                    insights.append(f"{category}: {stats['churn_rate']:.1%}")
        
        return f"Demographic insights: {', '.join(insights)}."


class DocumentContextProvider:
    """Provides context from uploaded documents."""
    
    def __init__(self):
        pass
    
    async def get_context(self, query: str) -> List[ContextPiece]:
        """Get relevant document context for the query."""
        context_pieces = []
        
        try:
            # Search documents based on query
            relevant_documents = document_manager.search_documents(query, limit=5)
            
            for doc in relevant_documents:
                # Determine relevance score based on category and tags
                relevance_score = self._calculate_relevance(query, doc)
                
                # Create context piece from document
                context_pieces.append(ContextPiece(
                    source=ContextSource.DOCUMENT_CONTENT,
                    content=f"Document '{doc.title or doc.original_filename}' "
                           f"(Category: {doc.category.value}) "
                           f"contains relevant information. "
                           f"Tags: {', '.join(doc.tags) if doc.tags else 'None'}.",
                    metadata={
                        'document_id': doc.document_id,
                        'category': doc.category.value,
                        'tags': doc.tags,
                        'title': doc.title
                    },
                    relevance_score=relevance_score,
                    source_details={
                        'document_id': doc.document_id,
                        'filename': doc.original_filename,
                        'category': doc.category.value,
                        'upload_date': doc.upload_timestamp
                    }
                ))
            
            # If no documents found, indicate this
            if not context_pieces:
                context_pieces.append(ContextPiece(
                    source=ContextSource.DOCUMENT_CONTENT,
                    content="No relevant documents found in the knowledge base for this query.",
                    metadata={'type': 'no_documents'},
                    relevance_score=0.0,
                    source_details={'search_query': query, 'results_count': 0}
                ))
        
        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            context_pieces.append(ContextPiece(
                source=ContextSource.DOCUMENT_CONTENT,
                content="Document search is currently unavailable.",
                metadata={'type': 'error', 'error': str(e)},
                relevance_score=0.1,
                source_details={'error': str(e)}
            ))
        
        return context_pieces
    
    def _calculate_relevance(self, query: str, document) -> float:
        """Calculate relevance score for a document given a query."""
        score = 0.5  # Base score
        
        query_lower = query.lower()
        
        # Boost score based on document category relevance
        category_boosts = {
            'churn_analysis': 0.3 if any(word in query_lower 
                                       for word in ['churn', 'retention', 'attrition']) else 0,
            'customer_feedback': 0.2 if any(word in query_lower 
                                          for word in ['feedback', 'satisfaction', 'customer']) else 0,
            'market_research': 0.2 if any(word in query_lower 
                                        for word in ['market', 'research', 'industry']) else 0,
            'telco_analysis': 0.2 if any(word in query_lower 
                                       for word in ['telco', 'telecom', 'fiber', 'broadband']) else 0
        }
        
        score += category_boosts.get(document.category.value, 0)
        
        # Boost score based on tag matches
        if document.tags:
            tag_matches = sum(1 for tag in document.tags if tag.lower() in query_lower)
            score += min(tag_matches * 0.1, 0.3)  # Max 0.3 boost from tags
        
        # Boost score based on title matches
        if document.title:
            title_matches = sum(1 for word in query_lower.split() 
                              if word in document.title.lower())
            score += min(title_matches * 0.05, 0.2)  # Max 0.2 boost from title
        
        return min(score, 1.0)  # Cap at 1.0


class UnifiedRAGSystem:
    """Main unified RAG system that combines dashboard and document context."""
    
    def __init__(self):
        self.query_classifier = QueryClassifier()
        self.dashboard_provider = DashboardContextProvider()
        self.document_provider = DocumentContextProvider()
        
        # Response templates
        self.response_templates = {
            QueryType.DASHBOARD_ONLY: self._generate_dashboard_response,
            QueryType.DOCUMENTS_ONLY: self._generate_document_response,
            QueryType.HYBRID: self._generate_hybrid_response,
            QueryType.UNKNOWN: self._generate_unknown_response
        }
    
    async def query_with_context(self,
                                question: str,
                                include_dashboard: bool = True,
                                include_documents: bool = True) -> UnifiedResponse:
        """
        Process a query using both dashboard and document context.
        
        Args:
            question: User's question
            include_dashboard: Whether to include dashboard analytics
            include_documents: Whether to include document content
            
        Returns:
            Unified response with context from multiple sources
        """
        start_time = datetime.now()
        query_id = f"query_{int(start_time.timestamp())}"
        
        try:
            # Classify the query
            query_type = self.query_classifier.classify_query(question)
            
            # Collect context based on query type and user preferences
            context_pieces = []
            sources_used = []
            
            # Get dashboard context
            if include_dashboard and query_type in [QueryType.DASHBOARD_ONLY, QueryType.HYBRID, QueryType.UNKNOWN]:
                dashboard_context = await self.dashboard_provider.get_context(question)
                context_pieces.extend(dashboard_context)
                if dashboard_context:
                    sources_used.append("dashboard_analytics")
            
            # Get document context
            if include_documents and query_type in [QueryType.DOCUMENTS_ONLY, QueryType.HYBRID, QueryType.UNKNOWN]:
                document_context = await self.document_provider.get_context(question)
                context_pieces.extend(document_context)
                if document_context and any(cp.relevance_score > 0 for cp in document_context):
                    sources_used.append("document_content")
            
            # Generate response using appropriate template
            response_generator = self.response_templates.get(query_type, self._generate_unknown_response)
            answer = response_generator(question, context_pieces)
            
            # Calculate confidence based on context quality
            confidence = self._calculate_confidence(context_pieces, query_type)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return UnifiedResponse(
                answer=answer,
                query_type=query_type,
                context_pieces=context_pieces,
                confidence=confidence,
                sources_used=sources_used,
                processing_time_ms=processing_time,
                query_id=query_id
            )
        
        except Exception as e:
            logger.error(f"Error in unified RAG query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return UnifiedResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                query_type=QueryType.UNKNOWN,
                context_pieces=[],
                confidence=0.0,
                sources_used=[],
                processing_time_ms=processing_time,
                query_id=query_id
            )
    
    def _generate_dashboard_response(self, question: str, context_pieces: List[ContextPiece]) -> str:
        """Generate response using primarily dashboard context."""
        dashboard_pieces = [cp for cp in context_pieces 
                           if cp.source == ContextSource.DASHBOARD_ANALYTICS]
        
        if not dashboard_pieces:
            return "I don't have sufficient dashboard data to answer your question."
        
        # Combine dashboard insights
        insights = [cp.content for cp in dashboard_pieces if cp.relevance_score > 0.5]
        
        response = f"Based on our current dashboard analytics:\n\n"
        response += "\n".join(f"• {insight}" for insight in insights)
        
        # Add data freshness note
        response += f"\n\nThis analysis is based on real-time dashboard data as of {datetime.now().strftime('%Y-%m-%d %H:%M')}."
        
        return response
    
    def _generate_document_response(self, question: str, context_pieces: List[ContextPiece]) -> str:
        """Generate response using primarily document context."""
        document_pieces = [cp for cp in context_pieces 
                          if cp.source == ContextSource.DOCUMENT_CONTENT]
        
        if not document_pieces or all(cp.relevance_score == 0 for cp in document_pieces):
            return "I don't have relevant documents in the knowledge base to answer your question."
        
        # Get relevant documents
        relevant_docs = [cp for cp in document_pieces if cp.relevance_score > 0]
        
        if not relevant_docs:
            return "No relevant documents were found for your question."
        
        response = f"Based on the documents in our knowledge base:\n\n"
        
        for i, doc_piece in enumerate(relevant_docs[:3], 1):  # Limit to top 3
            doc_info = doc_piece.source_details
            response += f"{i}. Document: '{doc_info.get('filename', 'Unknown')}'\n"
            response += f"   Category: {doc_info.get('category', 'Unknown')}\n"
            response += f"   Relevance: {doc_piece.relevance_score:.1%}\n\n"
        
        response += "Please note that responses are based on uploaded documents and may not reflect the most current data."
        
        return response
    
    def _generate_hybrid_response(self, question: str, context_pieces: List[ContextPiece]) -> str:
        """Generate response combining dashboard and document context."""
        dashboard_pieces = [cp for cp in context_pieces 
                           if cp.source == ContextSource.DASHBOARD_ANALYTICS and cp.relevance_score > 0.5]
        document_pieces = [cp for cp in context_pieces 
                          if cp.source == ContextSource.DOCUMENT_CONTENT and cp.relevance_score > 0]
        
        response = "Based on both our dashboard analytics and document insights:\n\n"
        
        # Dashboard insights section
        if dashboard_pieces:
            response += "**Current Analytics Data:**\n"
            for piece in dashboard_pieces:
                response += f"• {piece.content}\n"
            response += "\n"
        
        # Document insights section
        if document_pieces:
            response += "**Document Insights:**\n"
            for piece in document_pieces[:2]:  # Limit to top 2 documents
                doc_info = piece.source_details
                response += f"• {doc_info.get('filename', 'Document')}: {piece.content}\n"
            response += "\n"
        
        # Synthesis
        if dashboard_pieces and document_pieces:
            response += "**Synthesis:**\n"
            response += "This analysis combines real-time operational data with insights from uploaded research and documentation, "
            response += "providing both current performance metrics and contextual understanding from industry knowledge.\n"
        elif dashboard_pieces:
            response += "This analysis is based primarily on current dashboard data. "
            response += "Additional document insights were not available for this query.\n"
        elif document_pieces:
            response += "This analysis is based primarily on document insights. "
            response += "Current dashboard data was not available for this query.\n"
        else:
            response = "I don't have sufficient information from either dashboard analytics or documents to answer your question comprehensively."
        
        return response
    
    def _generate_unknown_response(self, question: str, context_pieces: List[ContextPiece]) -> str:
        """Generate response for unknown query types."""
        if not context_pieces:
            return ("I'm not sure how to best answer your question. Could you please rephrase it or "
                   "be more specific about whether you're looking for current analytics data or "
                   "information from uploaded documents?")
        
        # Use whatever context we have
        response = "Based on available information:\n\n"
        
        relevant_pieces = [cp for cp in context_pieces if cp.relevance_score > 0.3]
        if relevant_pieces:
            for piece in relevant_pieces[:3]:
                response += f"• {piece.content}\n"
        else:
            response += "Limited relevant information was found for your query."
        
        return response
    
    def _calculate_confidence(self, context_pieces: List[ContextPiece], query_type: QueryType) -> float:
        """Calculate confidence score based on context quality and query type."""
        if not context_pieces:
            return 0.0
        
        # Base confidence from relevance scores
        avg_relevance = sum(cp.relevance_score for cp in context_pieces) / len(context_pieces)
        
        # Boost confidence based on query type appropriateness
        type_boosts = {
            QueryType.DASHBOARD_ONLY: 0.1 if any(cp.source == ContextSource.DASHBOARD_ANALYTICS 
                                                 for cp in context_pieces) else -0.2,
            QueryType.DOCUMENTS_ONLY: 0.1 if any(cp.source == ContextSource.DOCUMENT_CONTENT 
                                                 for cp in context_pieces) else -0.2,
            QueryType.HYBRID: 0.2 if (any(cp.source == ContextSource.DASHBOARD_ANALYTICS for cp in context_pieces) and
                                     any(cp.source == ContextSource.DOCUMENT_CONTENT for cp in context_pieces)) else -0.1,
            QueryType.UNKNOWN: -0.1
        }
        
        confidence = avg_relevance + type_boosts.get(query_type, 0)
        
        # Number of sources bonus
        unique_sources = len(set(cp.source for cp in context_pieces))
        confidence += min(unique_sources * 0.05, 0.15)
        
        return max(0.0, min(1.0, confidence))


# Global unified RAG system instance
unified_rag_system = UnifiedRAGSystem()