import axios, { type AxiosInstance, AxiosError } from 'axios';
import { config } from '../utils/config';
import type {
  ApiResponse,
  Customer,
  CustomerFilters,
  PaginationParams,
  PaginatedResponse,
  // ChurnOverview,
  // ChurnByTenure,  
  // ChurnByContract,
  // ChurnByService,
  // DemographicAnalysis,
  // FinancialMetrics,
  PredictionRequest,
  PredictionResponse,
  ModelStatus,
  DocumentMetadata,
  DocumentUploadResponse,
  DocumentListResponse,
  ChatSession,
  ChatSessionRequest,
  ChatSessionResponse,
  ChatMessageRequest,
  ChatMessageResponse,
  HealthStatus,
  ServiceHealth,
  ApiError,
} from '../types/api';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: config.api_base_url,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add authentication token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        // Handle common error cases
        if (error.response?.status === 401) {
          // Unauthorized - clear token and redirect to login
          localStorage.removeItem('auth_token');
          // You might want to redirect to login page here
        }
        
        return Promise.reject(this.formatError(error));
      }
    );
  }

  private formatError(error: AxiosError): ApiError {
    return {
      code: error.response?.status?.toString() || 'NETWORK_ERROR',
      message: (error.response?.data as any)?.message || error.message || 'An error occurred',
      details: (error.response?.data as any)?.details || {},
      timestamp: new Date().toISOString(),
    };
  }

  // Customer endpoints
  async getCustomers(
    filters?: CustomerFilters,
    pagination?: PaginationParams
  ): Promise<PaginatedResponse<Customer>> {
    const response = await this.client.get('/customers', {
      params: { ...filters, ...pagination },
    });
    return response.data;
  }

  async getCustomer(id: string): Promise<Customer> {
    const response = await this.client.get(`/customers/${id}`);
    return response.data;
  }

  async createCustomer(customer: Partial<Customer>): Promise<Customer> {
    const response = await this.client.post('/customers', customer);
    return response.data;
  }

  async updateCustomer(id: string, customer: Partial<Customer>): Promise<Customer> {
    const response = await this.client.put(`/customers/${id}`, customer);
    return response.data;
  }

  async deleteCustomer(id: string): Promise<void> {
    await this.client.delete(`/customers/${id}`);
  }

  // Analytics endpoints
  async getChurnOverview(): Promise<any> {
    const response = await this.client.get('/analytics/churn-rate');
    return response.data;
  }

  async getChurnByTenure(): Promise<any> {
    const response = await this.client.get('/analytics/demographics');
    return response.data;
  }

  async getChurnByContract(): Promise<any> {
    const response = await this.client.get('/analytics/services');
    return response.data;
  }

  async getChurnByService(): Promise<any> {
    const response = await this.client.get('/analytics/services');
    return response.data;
  }

  async getDemographicAnalysis(): Promise<any> {
    const response = await this.client.get('/analytics/demographics');
    return response.data;
  }

  async getFinancialMetrics(): Promise<any> {
    const response = await this.client.get('/analytics/financial');
    return response.data;
  }

  // ML endpoints
  async predictChurn(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await this.client.post('/ml/predict', request);
    return response.data;
  }

  async getModelStatus(): Promise<ModelStatus> {
    const response = await this.client.get('/ml/model-status');
    return response.data;
  }

  async triggerModelRetrain(): Promise<ApiResponse> {
    const response = await this.client.post('/ml/retrain');
    return response.data;
  }

  // Document/RAG endpoints
  async uploadDocument(
    file: File,
    metadata: {
      category?: string;
      title?: string;
      description?: string;
      tags?: string;
      author?: string;
    }
  ): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (metadata.category) formData.append('category', metadata.category);
    if (metadata.title) formData.append('title', metadata.title);
    if (metadata.description) formData.append('description', metadata.description);
    if (metadata.tags) formData.append('tags', metadata.tags);
    if (metadata.author) formData.append('author', metadata.author);

    const response = await this.client.post('/rag/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async getDocuments(
    category?: string,
    tags?: string,
    limit?: number
  ): Promise<DocumentListResponse> {
    const response = await this.client.get('/rag/documents', {
      params: { category, tags, limit },
    });
    return response.data;
  }

  async getDocument(id: string): Promise<DocumentMetadata> {
    const response = await this.client.get(`/rag/documents/${id}`);
    return response.data;
  }

  async deleteDocument(id: string): Promise<ApiResponse> {
    const response = await this.client.delete(`/rag/documents/${id}`);
    return response.data;
  }

  // Chat endpoints
  async createChatSession(request: ChatSessionRequest): Promise<ChatSessionResponse> {
    const response = await this.client.post('/rag/chat/session', request);
    return response.data;
  }

  async getChatSession(sessionId: string): Promise<{
    success: boolean;
    session: ChatSession;
    conversation_history: any[];
    context: any;
  }> {
    const response = await this.client.get(`/rag/chat/session/${sessionId}`);
    return response.data;
  }

  async deleteChatSession(sessionId: string): Promise<ApiResponse> {
    const response = await this.client.delete(`/rag/chat/session/${sessionId}`);
    return response.data;
  }

  async sendChatMessage(request: ChatMessageRequest): Promise<ChatMessageResponse> {
    const response = await this.client.post('/rag/chat', request);
    return response.data;
  }

  async getUserSessions(userId: string): Promise<{
    success: boolean;
    user_id: string;
    sessions: ChatSession[];
    total_count: number;
  }> {
    const response = await this.client.get(`/rag/chat/sessions?user_id=${userId}`);
    return response.data;
  }

  // Health check endpoints
  async getHealthStatus(): Promise<HealthStatus> {
    const response = await this.client.get('/health');
    return response.data;
  }

  async getServiceHealth(): Promise<ServiceHealth> {
    const response = await this.client.get('/health/services');
    return response.data;
  }

  async getChatHealth(): Promise<HealthStatus> {
    const response = await this.client.get('/rag/chat/health');
    return response.data;
  }

  // Utility methods
  getStreamingUrl(endpoint: string): string {
    return `${config.api_base_url}${endpoint}`;
  }

  getWebSocketUrl(endpoint: string): string {
    return `${config.websocket_url}${endpoint}`;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
export default apiClient;