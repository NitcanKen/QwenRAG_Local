// Dashboard-specific type definitions
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