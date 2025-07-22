# Telco Customer Churn Dashboard with RAG - Implementation Plan

## Project Overview

### **Vision**

Build a comprehensive AI-powered telco customer churn dashboard that combines real-time analytics, machine learning predictions, and RAG (Retrieval-Augmented Generation) capabilities for intelligent document-based insights.

### **Core Capabilities**

- **Real-time Churn Analytics**: Interactive dashboard showing customer churn patterns, demographics, service impacts, and financial metrics
- **ML-Powered Predictions**: Random Forest model for churn prediction with automatic retraining
- **RAG Integration**: Chat interface allowing analysts to query both dashboard data and uploaded documents
- **Live Data Sync**: Automatic updates from Supabase within 5 minutes

---

## Tech Stack

### **Frontend**

- **Framework**: React 18+ with TypeScript
- **Styling**: Tailwind CSS
- **Charts/Visualization**: Recharts or Chart.js
- **State Management**: React Query + Zustand
- **Real-time Updates**: WebSocket or Server-Sent Events

### **Backend**

- **API Server**: FastAPI (Python)
- **Database**: Supabase (PostgreSQL)
- **Vector Database**: Qdrant (existing)
- **ML Pipeline**: scikit-learn (Random Forest)
- **RAG System**: LangChain + Ollama (existing)

### **Infrastructure**

- **Deployment**: Local Docker Compose
- **Background Jobs**: Celery + Redis
- **File Storage**: Local file system
- **Real-time**: WebSocket (FastAPI WebSocket)

---

## Implementation Stages

## Stage 1: Database & Infrastructure Setup

**Status**: ✅ Completed

### 1.1 Supabase Schema Design

**Dependencies**: None  
**Status**: ✅ Completed

**Tasks**:

- Create `customers` table matching CSV structure
- Set up proper indexes for performance
- Configure Row Level Security (RLS) policies
- Set up database triggers for change detection

**Database Schema**:

```sql
CREATE TABLE customers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id VARCHAR UNIQUE NOT NULL,
    gender VARCHAR,
    senior_citizen INTEGER,
    partner VARCHAR,
    dependents VARCHAR,
    tenure INTEGER,
    phone_service VARCHAR,
    multiple_lines VARCHAR,
    internet_service VARCHAR,
    online_security VARCHAR,
    online_backup VARCHAR,
    device_protection VARCHAR,
    tech_support VARCHAR,
    streaming_tv VARCHAR,
    streaming_movies VARCHAR,
    contract VARCHAR,
    paperless_billing VARCHAR,
    payment_method VARCHAR,
    monthly_charges DECIMAL,
    total_charges DECIMAL,
    churn INTEGER,
    tenure_group VARCHAR,
    monthly_charges_group VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE data_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR NOT NULL,
    operation VARCHAR NOT NULL,
    changed_at TIMESTAMP DEFAULT NOW()
);
```

### 1.2 Docker Environment Setup

**Dependencies**: 1.1 completed  
**Status**: ✅ Completed

**Tasks**:

- Create docker-compose.yml for all services
- Configure Qdrant container
- Set up Redis for background jobs
- Configure shared volumes and networks

**Docker Services**:

- Frontend (React app)
- Backend API (FastAPI)
- Qdrant (vector database)
- Redis (job queue)
- Celery worker (background tasks)

### 1.3 Data Migration

**Dependencies**: 1.1, 1.2 completed  
**Status**: ✅ Completed

**Tasks**:

- Import CSV data into Supabase
- Validate data integrity
- Set up initial aggregations
- Test database performance

---

## Stage 2: Backend API Development

**Status**: ⏳ Pending

### 2.1 Core API Structure

**Dependencies**: Stage 1 completed  
**Status**: ✅ Completed

**Tasks**:

- Set up FastAPI project structure
- Configure database connections
- Implement basic CRUD operations
- Set up error handling and logging

**API Endpoints Structure**:

```
/api/v1/
├── customers/
│   ├── GET /              # List customers with filters
│   ├── GET /{id}          # Get customer details
│   ├── POST /             # Create new customer
│   └── PUT /{id}          # Update customer
├── analytics/
│   ├── GET /churn-rate    # Overall churn metrics
│   ├── GET /demographics  # Demographic analysis
│   ├── GET /services      # Service impact analysis
│   └── GET /financial     # Financial metrics
├── ml/
│   ├── POST /predict      # Single customer prediction
│   ├── GET /model-status  # Model performance metrics
│   └── POST /retrain      # Trigger model retraining
└── rag/
    ├── POST /upload       # Upload documents
    ├── POST /chat         # Chat with dashboard + docs
    └── GET /documents     # List uploaded documents
```

### 2.2 Analytics Engine

**Dependencies**: 2.1 completed  
**Status**: ✅ Completed

**Tasks**:

- Implement churn rate calculations by various dimensions
- Create demographic analysis functions
- Build service impact analysis
- Develop financial metrics calculations
- Add caching for performance

**Key Analytics Functions**:

```python
# Churn Analysis
def get_churn_rate_by_tenure() -> Dict
def get_churn_rate_by_contract() -> Dict
def get_churn_rate_by_payment_method() -> Dict

# Customer Segmentation
def get_churn_by_demographics() -> Dict
def get_churn_by_services() -> Dict

# Financial Analysis
def get_charges_distribution() -> Dict
def get_revenue_impact() -> Dict
```

### 2.3 Real-time Data Sync

**Dependencies**: 2.2 completed  
**Status**: ✅ Completed

**Tasks**:

- Implement Supabase change detection
- Set up WebSocket server for real-time updates
- Create background job for data aggregation
- Add 5-minute sync mechanism

**Components**:

- Supabase webhook handler
- WebSocket connection manager
- Background aggregation jobs
- Change notification system

---

## Stage 3: Machine Learning Pipeline

**Status**: ✅ Completed

### 3.1 Model Development

**Dependencies**: Stage 2 completed  
**Status**: ✅ Completed

**Tasks**:

- Implement Random Forest churn prediction model
- Create feature engineering pipeline
- Set up model training and validation
- Implement model serialization

**ML Pipeline Components**:

```python
class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.preprocessor = FeaturePreprocessor()

    def train(self, data: pd.DataFrame) -> Dict
    def predict(self, customer_data: Dict) -> Dict
    def evaluate(self) -> Dict
    def save_model(self) -> str
    def load_model(self, path: str) -> None
```

### 3.2 Auto-Retraining System

**Dependencies**: 3.1 completed  
**Status**: ✅ Completed

**Tasks**:

- Implement model performance monitoring
- Create automatic retraining triggers
- Set up model comparison and selection
- Add model versioning

**Auto-Retraining Logic**:

- Monitor model performance metrics
- Trigger retraining when new data threshold reached
- Compare new model vs current model
- Deploy better performing model
- Keep model history and rollback capability

### 3.3 Prediction API

**Dependencies**: 3.2 completed  
**Status**: ✅ Completed

**Tasks**:

- Create prediction endpoints
- Implement batch prediction capability
- Add confidence scoring
- Set up prediction caching

---

## Stage 4: RAG System Enhancement

**Status**: ✅ Completed

### 4.1 Document Management

**Dependencies**: Stage 2 completed  
**Status**: ✅ Completed

**Tasks**:

- Extend existing RAG system for document upload
- Implement document categorization
- Add document metadata tracking
- Set up document preprocessing pipeline

**Document Types Support**:

- Industry reports (PDF)
- Customer feedback documents
- Market research reports
- Competitor analysis
- Internal strategy documents

### 4.2 Dashboard-Document Integration

**Dependencies**: 4.1, Stage 3 completed  
**Status**: ✅ Completed

**Tasks**:

- Create unified context system
- Implement dashboard data + document querying
- Add intelligent routing between data and documents
- Set up context-aware response generation

**Integration Features**:

```python
class UnifiedRAGSystem:
    def query_with_context(self,
        question: str,
        include_dashboard: bool = True,
        include_documents: bool = True
    ) -> Dict

    def get_dashboard_context(self, question: str) -> str
    def get_document_context(self, question: str) -> str
    def generate_integrated_response(self, contexts: List[str]) -> str
```

### 4.3 Chat Interface API

**Dependencies**: 4.2 completed  
**Status**: ✅ Completed

**Tasks**:

- Create chat endpoint with session management
- Implement conversation history
- Add streaming responses
- Set up context persistence

**Implementation Details**:

- ✅ Created comprehensive chat session management with Redis persistence
- ✅ Implemented DeepSeek-R1:8b integration service for advanced reasoning
- ✅ Built multiple chat interfaces: REST API, Server-Sent Events (SSE), and WebSocket
- ✅ Integrated with existing UnifiedRAGSystem for dashboard+document context
- ✅ Added conversation history, context persistence, and session cleanup
- ✅ Created comprehensive test suite for full integration validation

**Key Features Delivered**:

- Session-based chat management with Redis backend
- Streaming responses using DeepSeek-R1:8b model via Ollama
- Real-time WebSocket support for bidirectional communication
- Context-aware responses combining dashboard analytics and documents
- Conversation memory and session persistence
- Health monitoring and error handling
- Comprehensive API endpoints for session management

---

## Stage 5: Frontend Development

**Status**: ⏳ Pending

### 5.1 React Application Setup

**Dependencies**: None (can start in parallel)  
**Status**: ✅ Completed

**Tasks**:

- Initialize React + TypeScript project
- Configure Tailwind CSS
- Set up project structure and routing
- Configure development environment

**Implementation Details**:

- ✅ Created React + TypeScript project with Vite for fast development
- ✅ Configured Tailwind CSS with custom theme and utility classes
- ✅ Set up React Router for navigation between Dashboard, Analytics, and Chat pages
- ✅ Created comprehensive TypeScript types for all API responses
- ✅ Built layout components with responsive sidebar and header
- ✅ Configured environment variables and API client with axios and React Query
- ✅ Created placeholder pages with professional UI design
- ✅ Tested application startup and basic functionality

**Key Features Delivered**:

- Professional responsive layout with sidebar navigation
- Three main pages: Dashboard, Analytics, and AI Chat
- Comprehensive TypeScript type definitions
- API client setup with error handling
- Tailwind CSS theme with custom components
- React Query for state management
- Environment configuration for development/production

**Project Structure**:

```
src/
├── components/
│   ├── dashboard/
│   ├── charts/
│   ├── chat/
│   └── common/
├── pages/
│   ├── Dashboard.tsx
│   ├── Analytics.tsx
│   └── Chat.tsx
├── hooks/
├── services/
├── types/
└── utils/
```

### 5.2 Dashboard Components

**Dependencies**: 5.1, Stage 2 completed  
**Status**: ⏳ Completed

**Tasks**:

- Implement KPI cards component
- Create churn analysis charts
- Build customer segmentation visualizations
- Develop service analysis components
- Add financial metrics displays

**Key Components**:

```typescript
// Dashboard Components
<ChurnOverview />
<ChurnByTenure />
<ChurnByContract />
<ChurnByPaymentMethod />
<DemographicAnalysis />
<ServiceImpactAnalysis />
<FinancialMetrics />
<PredictiveAnalytics />
```

### 5.3 Real-time Updates

**Dependencies**: 5.2, Stage 2.3 completed  
**Status**: ⏳ Pending

**Tasks**:

- Implement WebSocket client
- Add real-time data updates
- Create loading states and error handling
- Set up automatic reconnection

### 5.4 Chat Interface

**Dependencies**: 5.1, Stage 4 completed  
**Status**: ⏳ Pending

**Tasks**:

- Build chat UI component
- Implement document upload interface
- Add conversation history
- Create response streaming

**Chat Features**:

- Document upload with progress
- Message history persistence
- Streaming responses
- Context indicators (dashboard vs documents)
- Export conversation capability

---

## Stage 6: Integration & Testing

**Status**: ⏳ Pending

### 6.1 API Integration

**Dependencies**: Stage 5.2, Stage 2 completed  
**Status**: ⏳ Pending

**Tasks**:

- Connect frontend to backend APIs
- Implement error handling
- Add loading states
- Set up data fetching hooks

### 6.2 End-to-End Testing

**Dependencies**: 6.1 completed  
**Status**: ⏳ Pending

**Tasks**:

- Test complete user workflows
- Validate data accuracy
- Test real-time updates
- Verify ML predictions
- Test RAG responses

### 6.3 Performance Optimization

**Dependencies**: 6.2 completed  
**Status**: ⏳ Pending

**Tasks**:

- Optimize dashboard loading times
- Implement efficient caching
- Optimize database queries
- Add progressive loading

---

## Stage 7: Production Readiness

**Status**: ⏳ Pending

### 7.1 Error Handling & Monitoring

**Dependencies**: Stage 6 completed  
**Status**: ⏳ Pending

**Tasks**:

- Implement comprehensive error handling
- Add logging and monitoring
- Set up health checks
- Create error recovery mechanisms

### 7.2 Documentation

**Dependencies**: 7.1 completed  
**Status**: ⏳ Pending

**Tasks**:

- Create API documentation
- Write user guide
- Document deployment process
- Create troubleshooting guide

### 7.3 Final Deployment Setup

**Dependencies**: 7.2 completed  
**Status**: ⏳ Pending

**Tasks**:

- Finalize Docker configuration
- Set up environment variables
- Create deployment scripts
- Test complete deployment process

---

## Key Features Specification

### **Dashboard Analytics**

#### Churn Analysis

- **Churn Rate by Tenure**: Bar chart showing churn percentage across tenure groups
- **Churn Rate by Contract**: Comparison across Month-to-month, One year, Two year
- **Churn Rate by Payment Method**: Analysis across Electronic check, Bank transfer, etc.

#### Customer Segmentation

- **Demographics**: Gender-based churn analysis
- **Senior Citizens**: Churn comparison between seniors and non-seniors
- **Family Status**: Partner and dependents impact on churn

#### Service Analysis

- **Internet Service Impact**: Churn rates for DSL, Fiber optic, No service
- **Add-on Services**: Impact of OnlineSecurity, OnlineBackup, etc. on retention
- **Streaming Services**: StreamingTV and StreamingMovies impact

#### Financial Metrics

- **Monthly Charges vs Churn**: Distribution analysis across charge groups
- **Total Charges**: Spending pattern analysis
- **Revenue Impact**: Financial impact of churn

#### Predictive Analytics

- **Churn Risk Scoring**: High/Medium/Low risk customer classification
- **Model Performance**: Accuracy, precision, recall metrics
- **Feature Importance**: Key drivers of churn

### **RAG Capabilities**

#### Document Support

- PDF industry reports
- Customer feedback documents
- Market research reports
- Competitor analysis
- Internal strategy documents

#### Intelligent Querying

- **Dashboard + Documents**: "Why are fiber optic customers churning according to market research?"
- **Contextual Responses**: Combine numerical data with document insights
- **Source Attribution**: Clear indication of information sources

#### Chat Features

- Conversation history
- Document upload with processing status
- Streaming responses
- Context switching between dashboard and documents

### **Real-time Features**

- 5-minute data refresh from Supabase
- Live dashboard updates
- WebSocket-based real-time notifications
- Background data processing

---

## Success Criteria

### **Functional Requirements**

- [ ] Dashboard displays all specified churn analytics
- [ ] ML model achieves >80% accuracy on churn prediction
- [ ] RAG system can answer questions using both dashboard and documents
- [ ] Real-time updates work within 5-minute requirement
- [ ] Auto-retraining improves or maintains model performance

### **Technical Requirements**

- [ ] Application runs locally via Docker Compose
- [ ] All components integrate successfully
- [ ] Error handling covers edge cases
- [ ] Performance meets analyst workflow needs
- [ ] Code is maintainable and well-documented

### **User Experience Requirements**

- [ ] Intuitive dashboard navigation
- [ ] Fast chart loading and interactions
- [ ] Smooth chat interface experience
- [ ] Clear data source indicators
- [ ] Responsive design for different screen sizes

---

## Dependencies Map

```
Stage 1 (Database) → Stage 2 (Backend API) → Stage 3 (ML Pipeline)
                                         ↓
Stage 5 (Frontend) ← Stage 4 (RAG Enhancement) ← Stage 2
                ↓
Stage 6 (Integration) → Stage 7 (Production)
```

## Status Legend

- ⏳ **Pending**: Not started
- 🔄 **In Progress**: Currently being worked on
- ✅ **Completed**: Finished and tested
- ❌ **Blocked**: Cannot proceed due to dependencies
- ⚠️ **Issues**: Has problems that need resolution

---

## Notes & Assumptions

### **Technical Assumptions**

- Ollama models are already set up and working
- Qdrant is available via Docker
- Local development environment has sufficient resources
- Supabase project is accessible

### **Data Assumptions**

- CSV data structure remains consistent
- New customer data follows same schema
- Data quality is maintained in Supabase
- Regular data updates will occur

### **Performance Assumptions**

- Dashboard serves analysts (not end customers)
- Query complexity is moderate
- Real-time updates are acceptable within 5 minutes
- Local deployment is sufficient for user load

---

_This plan serves as a living document and can be updated as requirements evolve or technical discoveries are made during implementation._
