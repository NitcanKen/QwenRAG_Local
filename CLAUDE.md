# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Feel free to use any mcp for your testing or any developing, especially the context7 mcp.
Check the venv / env we are working on, activate it and test it, make sure the pip install is working.
Everytime finish one stage (stage/substage), implement a test for it to make sure it works, only enough code to make that test pass.

## Project Documentation System

### 📋 plan.md vs 📚 memory.md - Key Distinctions

**plan.md** = ROADMAP & TASK TRACKER
- **Purpose**: Implementation roadmap and task tracking
- **Content**: Stages, dependencies, task lists, completion status
- **Focus**: WHAT needs to be done, WHEN, and in what ORDER
- **Usage**: Track progress, mark tasks as completed (✅), identify next steps
- **Structure**: Hierarchical stages → substages → tasks
- **Updates**: Mark status changes (⏳ Pending → ✅ Completed)

**memory.md** = TECHNICAL ARCHITECTURE DOCUMENTATION  
- **Purpose**: Comprehensive system knowledge base
- **Content**: Code structure, function analysis, dependencies, architecture
- **Focus**: HOW the system works, WHY design decisions were made
- **Usage**: Understand codebase, analyze relationships, guide future development
- **Structure**: Architecture → Files → Functions → Dependencies → Call Chains
- **Updates**: Add new implementations, document design decisions

### Memory Management Workflow
After completing each stage/substage:

1. **Update plan.md**: Mark the completed stage/substage as ✅ Completed
2. **Update memory.md**: Document the implementation details:
   - Record what files/folders were created or modified
   - Document main entry functions and their roles  
   - Identify key tool functions vs auxiliary functions
   - Annotate the importance and role of each function
   - Update call chain analysis with new function flows
   - Update inter-module dependencies analysis
   - Record performance characteristics and design decisions
   - Note any technical debt or future considerations
3. **Next Steps**: Review plan.md to identify the next unmarked task

### File Usage Guidelines:
- **Consult plan.md**: To understand project roadmap and find next tasks
- **Consult memory.md**: To understand existing code architecture before making changes
- **Update both files**: After completing any significant implementation work

CRITICAL: Never confuse these files - plan.md tracks PROGRESS, memory.md documents ARCHITECTURE.

Always follow the instructions. When I say "go", find the next unmarked task in plan.md and start do it.


## Available MCP Tools & Usage Guidelines

### Supabase MCP (`mcp__supabase__*`)
**Purpose**: Database operations, schema management, and backend development

**Key Tools**:
- `mcp__supabase__execute_sql`: Run SQL queries on the database
- `mcp__supabase__apply_migration`: Execute DDL operations and schema changes
- `mcp__supabase__list_tables`: View database structure
- `mcp__supabase__create_branch`: Create development branches for safe testing
- `mcp__supabase__generate_typescript_types`: Generate TypeScript types from schema
- `mcp__supabase__get_project_url` / `mcp__supabase__get_anon_key`: Get connection details

**When to Use**:
- Setting up the customer churn database schema (Stage 1.1)
- Creating and testing database migrations
- Querying customer data for analytics development
- Generating TypeScript types for frontend development
- Testing database changes safely with branches

### Playwright MCP (`mcp__playwright__*`)
**Purpose**: Browser automation, frontend testing, and UI validation

**Key Tools**:
- `mcp__playwright__browser_navigate`: Navigate to web pages
- `mcp__playwright__browser_snapshot`: Capture page state for analysis
- `mcp__playwright__browser_click` / `mcp__playwright__browser_type`: Interact with UI elements
- `mcp__playwright__browser_take_screenshot`: Visual testing and documentation
- `mcp__playwright__browser_tab_new` / `mcp__playwright__browser_tab_select`: Multi-tab testing

**When to Use**:
- Testing the React dashboard UI (Stage 5.2, 5.3)
- Validating real-time updates and WebSocket functionality
- End-to-end testing of the chat interface (Stage 5.4)
- Testing user workflows across dashboard and RAG features
- Capturing screenshots for documentation

### IDE MCP (`mcp__ide__*`)
**Purpose**: Code analysis and Python execution

**Key Tools**:
- `mcp__ide__getDiagnostics`: Check for TypeScript/Python errors
- `mcp__ide__executeCode`: Run Python code in Jupyter kernel

**When to Use**:
- Validating ML model development (Stage 3.1, 3.2)
- Testing data analysis and churn prediction algorithms
- Running analytics calculations during backend development
- Debugging Python code issues

### Context7 MCP (`mcp__context7__*`)
**Purpose**: Up-to-date library documentation and code examples

**Key Tools**:
- `mcp__context7__resolve-library-id`: Find the correct library ID for documentation lookup
- `mcp__context7__get-library-docs`: Retrieve comprehensive documentation, API references, and code examples

**When to Use**:
- Learning React, FastAPI, or other library APIs during development
- Finding best practices and code examples for frontend components (Stage 5.1, 5.2)
- Understanding TypeScript, Tailwind CSS, or visualization library APIs
- Getting up-to-date documentation for backend libraries (FastAPI, SQLAlchemy, etc.)
- Resolving implementation questions with authoritative library documentation

**Usage Pattern**:
1. First call `resolve-library-id` with library name (e.g., "react", "fastapi", "recharts")
2. Use the returned Context7-compatible ID with `get-library-docs` for detailed documentation
3. Focus searches with `topic` parameter for specific functionality (e.g., "hooks", "routing", "charts")

### Standard Tools (Non-MCP)
**File Operations**: `Read`, `Write`, `Edit`, `MultiEdit`, `Glob`, `Grep`
**System Operations**: `Bash`, `LS`
**Web Operations**: `WebFetch`, `WebSearch`
**Project Management**: `TodoWrite`

### MCP Priority Guidelines
1. **Use Supabase MCP first** for any database-related tasks
2. **Use Playwright MCP** for frontend testing and UI validation
3. **Use Context7 MCP** for authoritative library documentation and API references
4. **Prefer MCP tools** over standard tools when functionality overlaps
5. **Combine MCP tools** with standard tools for comprehensive workflows

### Development Workflow Integration
- **Stage 1 (Database Setup)**: Primarily Supabase MCP
- **Stage 2-3 (Backend/ML)**: Supabase MCP + IDE MCP + Context7 MCP + standard tools
- **Stage 5 (Frontend)**: Context7 MCP + Playwright MCP + standard file tools
- **Stage 6 (Testing)**: All MCP tools for comprehensive validation

## Project Overview

This project is building a comprehensive **Telco Customer Churn Dashboard** with AI-powered RAG capabilities. It combines real-time analytics, machine learning, and intelligent document querying into a unified intelligence platform for churn analysis and strategic decision-making.

### Vision
Create an analyst-focused dashboard that merges numerical churn analytics with contextual document knowledge, allowing queries like "Why are fiber optic customers churning according to uploaded market research?" while providing live data updates within 5 minutes.

## Architecture

### Tech Stack

#### Frontend
- **Framework**: React 18+ with TypeScript
- **Styling**: Tailwind CSS
- **Charts/Visualization**: Recharts or Chart.js
- **State Management**: React Query + Zustand
- **Real-time Updates**: WebSocket or Server-Sent Events

#### Backend
- **API Server**: FastAPI (Python)
- **Database**: Supabase (PostgreSQL)
- **Vector Database**: Qdrant (existing)
- **ML Pipeline**: scikit-learn (Random Forest)
- **RAG System**: LangChain + Ollama (existing)

#### Infrastructure
- **Deployment**: Local Docker Compose
- **Background Jobs**: Celery + Redis
- **File Storage**: Local file system
- **Real-time**: WebSocket (FastAPI WebSocket)

### Core Components
- **Analytics Dashboard**: Interactive churn visualizations by tenure, contract, demographics, services, financials
- **ML Prediction Engine**: Random Forest model with auto-retraining on new data
- **RAG Chat Interface**: Query both dashboard data and uploaded documents simultaneously
- **Real-time Data Sync**: 5-minute refresh from Supabase with WebSocket updates
- **Document Intelligence**: Process industry reports, market research, customer feedback

### Current Foundation
- **Existing RAG System**: `qwen_local_rag_agent.py` - Streamlit-based RAG with Ollama + Qdrant
- **LLM Models**: Qwen3, Gemma3, DeepSeek via Ollama
- **Vector Search**: Qdrant with similarity-based document retrieval
- **Document Processing**: PDF upload, web scraping, chunking

## Development Commands

### Current RAG System (Foundation)
```bash
# Activate virtual environment
QwanRAG\Scripts\activate

# Install current dependencies
pip install -r requirements.txt

# Start Qdrant vector database
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant

# Pull Ollama models
ollama pull qwen3:1.7b
ollama pull snowflake-arctic-embed

# Run current Streamlit RAG app
streamlit run qwen_local_rag_agent.py
```

### Target Dashboard System (To Be Built)
```bash
# Full system startup (when complete)
docker-compose up -d

# Individual service management
docker-compose up frontend backend qdrant redis celery

# Database migrations (Supabase)
# TBD: Migration scripts for customer data schema

# ML model training
# TBD: Python scripts for Random Forest training/retraining

# Development mode
npm run dev          # Frontend development server
uvicorn app:app --reload  # Backend development server
```

### Testing Commands
```bash
# Frontend tests (when implemented)
npm test
npm run test:e2e

# Backend tests (when implemented)  
pytest
pytest tests/test_ml_pipeline.py
pytest tests/test_rag_integration.py

# ML model validation (when implemented)
python scripts/validate_model.py
```

## Project Structure (Target Architecture)

```
QwenRAG_Local/
├── frontend/                 # React + TypeScript dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── dashboard/    # Analytics visualizations
│   │   │   ├── charts/       # Recharts components
│   │   │   ├── chat/         # RAG chat interface
│   │   │   └── common/       # Shared UI components
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx # Main analytics page
│   │   │   ├── Analytics.tsx # Detailed metrics
│   │   │   └── Chat.tsx      # RAG chat page
│   │   ├── hooks/            # React Query + custom hooks
│   │   ├── services/         # API clients
│   │   └── types/            # TypeScript definitions
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── api/
│   │   │   ├── analytics/    # Churn analytics endpoints
│   │   │   ├── ml/           # ML prediction endpoints
│   │   │   └── rag/          # RAG chat endpoints
│   │   ├── core/
│   │   │   ├── database.py   # Supabase connection
│   │   │   ├── ml_pipeline.py # Random Forest model
│   │   │   └── rag_system.py # Enhanced RAG integration
│   │   ├── models/           # Pydantic models
│   │   └── services/         # Business logic
├── data/
│   ├── csv/                  # Customer churn datasets
│   ├── migrations/           # Database schema scripts
│   └── ml_models/            # Trained model artifacts
├── docker/                   # Docker configurations
├── scripts/                  # Utility and setup scripts
├── qwen_local_rag_agent.py  # Current Streamlit RAG (foundation)
├── requirements.txt          # Current Python dependencies
├── docker-compose.yml        # Full system orchestration
└── README.md                 # Project documentation
```

## Key Data Structures

### Customer Database Schema (Supabase)
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

### Analytics API Endpoints
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

### RAG Integration Points
- **Dashboard Context**: Inject live analytics data into RAG queries
- **Document Types**: Industry reports, market research, customer feedback
- **Unified Querying**: "Why are fiber customers churning based on Q3 market report?"
- **Source Attribution**: Clear distinction between dashboard data vs documents

## Current Foundation (qwen_local_rag_agent.py)

### Existing Classes & Functions
- `OllamaEmbedderr`: Custom embeddings wrapper for LangChain
- `init_qdrant()`: Qdrant client initialization  
- `process_pdf(file)`, `process_web(url)`: Document processing
- `create_vector_store()`: Vector database management
- `get_rag_agent()`, `get_web_search_agent()`: Agent initialization

### RAG Configuration
- Collection: "test-qwen-r1" (1024 dimensions)
- Models: qwen3:1.7b (default), gemma3:1b/4b, deepseek-r1:1.5b
- Chunking: 1000 chars, 200 overlap
- Similarity threshold: 0.7 (adjustable)

## Implementation Stages

### Stage 1: Database & Infrastructure Setup
- **1.1 Supabase Schema Design**: Create customers table, indexes, RLS policies, triggers
- **1.2 Docker Environment Setup**: docker-compose.yml for all services
- **1.3 Data Migration**: Import CSV data, validate integrity

### Stage 2: Backend API Development  
- **2.1 Core API Structure**: FastAPI project, database connections, CRUD operations
- **2.2 Analytics Engine**: Churn calculations, demographic analysis, financial metrics
- **2.3 Real-time Data Sync**: Supabase webhooks, WebSocket server, 5-minute sync

### Stage 3: Machine Learning Pipeline
- **3.1 Model Development**: Random Forest implementation, feature engineering
- **3.2 Auto-Retraining System**: Performance monitoring, automatic triggers
- **3.3 Prediction API**: Prediction endpoints, batch processing, confidence scoring

### Stage 4: RAG System Enhancement
- **4.1 Document Management**: Extend existing RAG for uploads, categorization
- **4.2 Dashboard-Document Integration**: Unified context system, intelligent routing
- **4.3 Chat Interface API**: Session management, conversation history, streaming

### Stage 5: Frontend Development
- **5.1 React Application Setup**: TypeScript project, Tailwind CSS, routing
- **5.2 Dashboard Components**: KPI cards, churn charts, segmentation visualizations
- **5.3 Real-time Updates**: WebSocket client, live data updates
- **5.4 Chat Interface**: Chat UI, document upload, conversation history

### Stage 6: Integration & Testing
- **6.1 API Integration**: Connect frontend to backend APIs
- **6.2 End-to-End Testing**: Complete user workflows, data accuracy
- **6.3 Performance Optimization**: Loading times, caching, database queries

### Stage 7: Production Readiness
- **7.1 Error Handling & Monitoring**: Comprehensive error handling, logging
- **7.2 Documentation**: API docs, user guide, deployment process
- **7.3 Final Deployment Setup**: Docker configuration, environment variables

## Prerequisites

### Infrastructure
- Docker & Docker Compose
- Ollama with models: qwen3:1.7b, snowflake-arctic-embed
- Qdrant vector database (existing setup)
- Supabase project for customer data
- Node.js 18+ for React frontend
- Redis for background jobs
- Python 3.8+ environment

### Development Tools
- Existing requirements.txt foundation
- FastAPI for backend development
- React 18+ with TypeScript
- Tailwind CSS for styling
- Recharts or Chart.js for visualizations
- Optional: Exa API key for web search

## Key Analytics Features

### Dashboard Visualizations
- **Churn Analysis**: Rate by tenure, contract, payment method
- **Demographics**: Gender, senior citizens, family status impact
- **Service Analysis**: Internet service, add-ons, streaming services impact
- **Financial Metrics**: Monthly charges distribution, revenue impact
- **Predictive Analytics**: Risk scoring, model performance metrics

### ML Pipeline Components
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

### RAG Enhancement Features
- **Document Types**: Industry reports, market research, customer feedback
- **Unified Querying**: Combine dashboard data with document insights
- **Context Switching**: Clear source attribution (dashboard vs documents)
- **Conversation Memory**: Multi-turn analytics discussions

## ⚠️ Common Error Patterns & Prevention

### 1. TypeScript Import/Export Errors
**Symptoms**: `The requested module does not provide an export named 'X'`
**Root Cause**: Mixing API types with UI-specific types in same file
**Prevention**:
- Separate API client types from UI component types 
- Use dedicated files: `api.ts` for API, `dashboard.ts` for UI types
- Always verify export names match import statements exactly
**Fix Pattern**: Create separate type files and update all imports consistently

### 2. Backend Service Import Mismatches  
**Symptoms**: `cannot import name 'ServiceName'` 
**Root Cause**: Class names don't match between definition and import
**Prevention**:
- Always verify actual exported class names before importing
- Use consistent naming conventions across files
- Check class definitions when getting import errors
**Fix Pattern**: Update imports to match exact exported class names

### 3. Database Initialization Context Errors
**Symptoms**: `greenlet_spawn has not been called`, async context issues
**Root Cause**: Database operations called in wrong async context during startup
**Prevention**:
- Initialize database connections lazily, not during import
- Implement proper fallback mechanisms for external services
- Use dependency injection patterns for service initialization
**Fix Pattern**: Make service initialization lazy with graceful fallbacks

### 4. External Service Configuration Errors
**Symptoms**: `Invalid API key`, `Connection refused` for external services
**Root Cause**: Missing or invalid credentials for external services (Supabase, etc.)
**Prevention**:
- Always implement fallback mechanisms for external dependencies
- Use local alternatives (SQLite) when external services unavailable
- Validate credentials before making service calls
**Fix Pattern**: Graceful degradation with local fallbacks

### 5. Port Binding & Process Management
**Symptoms**: `Port already in use`, `WinError 10013` 
**Root Cause**: Previous server instances not properly terminated
**Prevention**:
- Always check for existing processes before starting servers
- Use consistent port management (8001 for backend, 3000 for frontend)
- Implement proper shutdown procedures
**Fix Pattern**: Kill existing processes and use alternative ports

### 6. Missing Dependencies 
**Symptoms**: `ModuleNotFoundError`, import errors for packages
**Root Cause**: Required packages not installed in current environment
**Prevention**:
- Always verify virtual environment is activated
- Run `pip install package_name` for missing dependencies
- Keep requirements.txt updated with all dependencies
**Fix Pattern**: Install missing packages immediately when encountered

### 7. Content Security Policy (CSP) Issues
**Symptoms**: External resources blocked, blank pages in browser tools
**Root Cause**: Overly restrictive CSP headers blocking necessary resources
**Prevention**:
- Implement context-aware CSP policies (different rules for API vs docs)
- Allow necessary CDN resources for development tools (Swagger UI)
- Test browser console for CSP violations
**Fix Pattern**: Update CSP middleware with appropriate resource allowlists

### 8. Async Database Driver Compatibility
**Symptoms**: Async operation errors with database drivers
**Root Cause**: Using sync drivers (`sqlite://`) with async operations
**Prevention**:
- Always use async-compatible drivers (`sqlite+aiosqlite://`)
- Match database URL schemes to async/sync requirements
- Install proper async database drivers (aiosqlite, asyncpg)
**Fix Pattern**: Update DATABASE_URL to use async-compatible drivers

### 🛡️ Error Prevention Checklist
Before starting any development work:

1. **Environment Check**:
   - ✅ Virtual environment activated
   - ✅ All required dependencies installed
   - ✅ No conflicting processes on target ports

2. **Service Dependencies**:
   - ✅ External services (Redis, Qdrant) running
   - ✅ Fallback mechanisms implemented
   - ✅ Configuration files properly set

3. **Code Structure**:
   - ✅ Import/export names match exactly
   - ✅ Type definitions properly separated
   - ✅ Async/sync patterns consistent

4. **Testing Strategy**:
   - ✅ Test each component in isolation
   - ✅ Verify integration points work
   - ✅ Check browser console for errors

**Remember**: When encountering errors, always implement the fix in a way that prevents the same issue from recurring. Add fallbacks, validation, and proper error handling.