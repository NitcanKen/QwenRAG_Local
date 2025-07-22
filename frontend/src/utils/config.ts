import type { AppConfig } from '../types/dashboard';

// Environment-based configuration
const getBaseURL = (): string => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  // Default to localhost in development
  if (import.meta.env.DEV) {
    return 'http://localhost:8000/api/v1';
  }
  
  // Production URL - adjust as needed
  return '/api/v1';
};

const getWebSocketURL = (): string => {
  if (import.meta.env.VITE_WS_BASE_URL) {
    return import.meta.env.VITE_WS_BASE_URL;
  }
  
  // Convert HTTP URL to WebSocket URL
  const baseURL = getBaseURL();
  return baseURL.replace(/^https?:/, 'ws:').replace('/api/v1', '/api/v1');
};

export const config: AppConfig = {
  api_base_url: getBaseURL(),
  websocket_url: getWebSocketURL(),
  features: {
    real_time_updates: true,
    chat_interface: true,
    document_upload: true,
    ml_predictions: true,
  },
  polling_intervals: {
    analytics: 30000, // 30 seconds
    health_check: 60000, // 1 minute
  },
};

// Environment variables with defaults
export const env = {
  isDev: import.meta.env.DEV,
  isProd: import.meta.env.PROD,
  apiBaseUrl: config.api_base_url,
  wsBaseUrl: config.websocket_url,
  
  // Feature flags
  enableRealTime: import.meta.env.VITE_ENABLE_REALTIME !== 'false',
  enableChat: import.meta.env.VITE_ENABLE_CHAT !== 'false',
  enableDocuments: import.meta.env.VITE_ENABLE_DOCUMENTS !== 'false',
  enableML: import.meta.env.VITE_ENABLE_ML !== 'false',
  
  // Debug flags
  debugApi: import.meta.env.VITE_DEBUG_API === 'true',
  debugWebSocket: import.meta.env.VITE_DEBUG_WS === 'true',
} as const;

// Validation
if (!config.api_base_url) {
  throw new Error('API base URL not configured');
}

console.log('App Configuration:', {
  environment: import.meta.env.MODE,
  apiBaseUrl: config.api_base_url,
  wsBaseUrl: config.websocket_url,
  features: config.features,
});