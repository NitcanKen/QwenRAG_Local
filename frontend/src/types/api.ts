// API Response Types for Telco Customer Churn Dashboard

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

// Customer Data Types
export interface Customer {
  id: string;
  customer_id: string;
  gender: string;
  senior_citizen: number;
  partner: string;
  dependents: string;
  tenure: number;
  phone_service: string;
  multiple_lines: string;
  internet_service: string;
  online_security: string;
  online_backup: string;
  device_protection: string;
  tech_support: string;
  streaming_tv: string;
  streaming_movies: string;
  contract: string;
  paperless_billing: string;
  payment_method: string;
  monthly_charges: number;
  total_charges: number;
  churn: number;
  tenure_group: string;
  monthly_charges_group: string;
  created_at: string;
  updated_at: string;
}

// Analytics Types
export interface ChurnOverview {
  total_customers: number;
  churned_customers: number;
  retained_customers: number;
  churn_rate: number;
  retention_rate: number;
  last_updated: string;
}

export interface ChurnByTenure {
  [tenureGroup: string]: {
    customer_count: number;
    churned_count: number;
    churn_rate: number;
  };
}

export interface ChurnByContract {
  [contractType: string]: {
    customer_count: number;
    churned_count: number;
    churn_rate: number;
  };
}

export interface ChurnByService {
  [serviceType: string]: {
    customer_count: number;
    churned_count: number;
    churn_rate: number;
  };
}

export interface DemographicAnalysis {
  gender_analysis: {
    [gender: string]: {
      customer_count: number;
      churned_count: number;
      churn_rate: number;
    };
  };
  senior_citizen_analysis: {
    [category: string]: {
      customer_count: number;
      churned_count: number;
      churn_rate: number;
    };
  };
  family_analysis: {
    [category: string]: {
      customer_count: number;
      churned_count: number;
      churn_rate: number;
    };
  };
}

export interface FinancialMetrics {
  charges_distribution: {
    [chargeGroup: string]: {
      customer_count: number;
      churned_count: number;
      churn_rate: number;
      avg_monthly_charges: number;
      avg_total_charges: number;
    };
  };
  revenue_impact: {
    total_revenue: number;
    lost_revenue: number;
    retained_revenue: number;
    avg_revenue_per_user: number;
  };
}

// ML Prediction Types
export interface PredictionRequest {
  customer_data: Partial<Customer>;
}

export interface PredictionResponse {
  prediction: number;
  probability: number;
  confidence: number;
  risk_level: 'low' | 'medium' | 'high';
  feature_importance: Record<string, number>;
}

export interface ModelStatus {
  model_version: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  last_trained: string;
  training_data_size: number;
  features_used: string[];
}

// RAG System Types
export interface DocumentMetadata {
  document_id: string;
  original_filename: string;
  title?: string;
  description?: string;
  category: DocumentCategory;
  tags: string[];
  author?: string;
  upload_timestamp: string;
  chunk_count: number;
  status: DocumentStatus;
}

export type DocumentCategory = 
  | 'industry_report'
  | 'customer_feedback'
  | 'market_research'
  | 'competitor_analysis'
  | 'strategy_document'
  | 'telco_analysis'
  | 'churn_analysis'
  | 'other';

export type DocumentStatus = 'processing' | 'completed' | 'failed';

export interface DocumentUploadResponse {
  success: boolean;
  document_id: string;
  filename: string;
  category: string;
  chunk_count: number;
  status: string;
  message: string;
}

export interface DocumentListResponse {
  success: boolean;
  documents: DocumentMetadata[];
  total_count: number;
  filters_applied: Record<string, any>;
}

// Chat System Types
export interface ChatSession {
  session_id: string;
  user_id?: string;
  created_at: string;
  updated_at: string;
  settings: ChatSessionSettings;
  message_count: number;
  is_active: boolean;
}

export interface ChatSessionSettings {
  model: string;
  temperature: number;
  max_tokens: number;
  top_p: number;
  include_dashboard: boolean;
  include_documents: boolean;
  stream_response: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
  sources?: ChatSource[];
  confidence?: number;
  processing_time_ms?: number;
}

export interface ChatSource {
  type: 'dashboard_analytics' | 'document_content';
  content_preview: string;
  relevance_score: number;
  metadata: Record<string, any>;
  endpoint?: string;
  document_id?: string;
  filename?: string;
  category?: string;
}

export interface ChatSessionRequest {
  user_id?: string;
  custom_settings?: Partial<ChatSessionSettings>;
}

export interface ChatSessionResponse {
  success: boolean;
  session_id: string;
  message: string;
  settings: ChatSessionSettings;
}

export interface ChatMessageRequest {
  message: string;
  session_id?: string;
  user_id?: string;
  include_dashboard?: boolean;
  include_documents?: boolean;
  stream_response?: boolean;
}

export interface ChatMessageResponse {
  success: boolean;
  session_id: string;
  message_id: string;
  response: string;
  sources: ChatSource[];
  metadata: Record<string, any>;
  timestamp: string;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: 'chat_message' | 'heartbeat' | 'update_settings' | 'connected' | 'message_received' | 'stream_chunk' | 'stream_complete' | 'error' | 'heartbeat_ack' | 'settings_updated';
  data?: any;
  timestamp?: string;
}

export interface StreamChunk {
  content: string;
  done: boolean;
  metadata?: Record<string, any>;
  timestamp: string;
}

// Real-time Types
export interface DataChange {
  id: string;
  table_name: string;
  operation: 'INSERT' | 'UPDATE' | 'DELETE';
  changed_at: string;
}

export interface RealtimeUpdate {
  type: 'data_change' | 'analytics_update' | 'model_update';
  payload: any;
  timestamp: string;
}

// Health Check Types
export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  components?: Record<string, any>;
  error?: string;
}

export interface ServiceHealth {
  database: HealthStatus;
  analytics: HealthStatus;
  ml_service: HealthStatus;
  rag_system: HealthStatus;
  chat_service?: HealthStatus;
}

// Query and Filter Types
export interface CustomerFilters {
  gender?: string[];
  senior_citizen?: number[];
  partner?: string[];
  dependents?: string[];
  tenure_min?: number;
  tenure_max?: number;
  contract?: string[];
  payment_method?: string[];
  monthly_charges_min?: number;
  monthly_charges_max?: number;
  internet_service?: string[];
  churn?: number[];
}

export interface PaginationParams {
  page: number;
  limit: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    pages: number;
    has_next: boolean;
    has_prev: boolean;
  };
}

// Error Types
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
}

export interface ValidationError {
  field: string;
  message: string;
  code: string;
}

// Configuration Types
export interface AppConfig {
  api_base_url: string;
  websocket_url: string;
  features: {
    real_time_updates: boolean;
    chat_interface: boolean;
    document_upload: boolean;
    ml_predictions: boolean;
  };
  polling_intervals: {
    analytics: number;
    health_check: number;
  };
}