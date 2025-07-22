import React from 'react';

// Import new components
import KPICards from '../components/dashboard/KPICards';
import ChurnByTenureChart from '../components/charts/ChurnByTenureChart';
import ChurnByContractChart from '../components/charts/ChurnByContractChart';
import ServiceAnalysis from '../components/dashboard/ServiceAnalysis';
import DemographicAnalysis from '../components/dashboard/DemographicAnalysis';
import FinancialMetrics from '../components/dashboard/FinancialMetrics';

// Mock data for demo
const mockChurnOverview = {
  total_customers: 7043,
  churned_customers: 1869,
  retained_customers: 5174,
  churn_rate: 0.265,
  retention_rate: 0.735,
  last_updated: new Date().toISOString(),
};

const mockTenureData = {
  '0-12 months': { customer_count: 1200, churned_count: 450, churn_rate: 0.375 },
  '13-24 months': { customer_count: 1800, churned_count: 520, churn_rate: 0.289 },
  '25-48 months': { customer_count: 2000, churned_count: 480, churn_rate: 0.240 },
  '49-72 months': { customer_count: 1500, churned_count: 300, churn_rate: 0.200 },
  '73+ months': { customer_count: 543, churned_count: 119, churn_rate: 0.219 },
};

const mockContractData = {
  'Month-to-month': { customer_count: 3875, churned_count: 1655, churn_rate: 0.427 },
  'One year': { customer_count: 1473, churned_count: 166, churn_rate: 0.113 },
  'Two year': { customer_count: 1695, churned_count: 48, churn_rate: 0.028 },
};

const mockServiceData = {
  'Fiber optic': { customer_count: 3096, churned_count: 1297, churn_rate: 0.419 },
  'DSL': { customer_count: 2421, churned_count: 459, churn_rate: 0.190 },
  'No': { customer_count: 1526, churned_count: 113, churn_rate: 0.074 },
};

const mockDemographicData = {
  gender_analysis: {
    'Male': { customer_count: 3555, churned_count: 930, churn_rate: 0.262 },
    'Female': { customer_count: 3488, churned_count: 939, churn_rate: 0.269 },
  },
  senior_citizen_analysis: {
    'Non-Senior': { customer_count: 5901, churned_count: 1393, churn_rate: 0.236 },
    'Senior': { customer_count: 1142, churned_count: 476, churn_rate: 0.417 },
  },
  family_analysis: {
    'Has Partner': { customer_count: 3402, churned_count: 710, churn_rate: 0.209 },
    'No Partner': { customer_count: 3641, churned_count: 1159, churn_rate: 0.318 },
    'Has Dependents': { customer_count: 2110, churned_count: 431, churn_rate: 0.204 },
    'No Dependents': { customer_count: 4933, churned_count: 1438, churn_rate: 0.291 },
  },
};

const mockFinancialData = {
  charges_distribution: {
    'Low (0-35)': { 
      customer_count: 1200, 
      churned_count: 220, 
      churn_rate: 0.183, 
      avg_monthly_charges: 25.5, 
      avg_total_charges: 1200.0 
    },
    'Medium (35-65)': { 
      customer_count: 2800, 
      churned_count: 680, 
      churn_rate: 0.243, 
      avg_monthly_charges: 50.2, 
      avg_total_charges: 2400.0 
    },
    'High (65+)': { 
      customer_count: 3043, 
      churned_count: 969, 
      churn_rate: 0.318, 
      avg_monthly_charges: 85.7, 
      avg_total_charges: 4200.0 
    },
  },
  revenue_impact: {
    total_revenue: 456789.50,
    lost_revenue: 121234.75,
    retained_revenue: 335554.75,
    avg_revenue_per_user: 64.9,
  },
};

const DashboardDemo: React.FC = () => {
  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header Section */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Customer Churn Dashboard - Demo
            </h1>
            <p className="text-gray-600 mt-2">
              Real-time analytics and insights into customer churn patterns (Demo with Mock Data)
            </p>
          </div>
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-orange-500 rounded-full mr-2"></div>
              Demo Mode
            </div>
            <span>•</span>
            <span>Mock Data</span>
          </div>
        </div>
      </div>

      {/* KPI Cards Section */}
      <KPICards 
        data={mockChurnOverview} 
        loading={false} 
        error={null} 
      />

      {/* Main Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <ChurnByTenureChart
          data={mockTenureData}
          loading={false}
          error={null}
        />
        <ChurnByContractChart
          data={mockContractData}
          loading={false}
          error={null}
        />
      </div>

      {/* Service Analysis and Demographics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <ServiceAnalysis
          data={mockServiceData}
          loading={false}
          error={null}
        />
        <DemographicAnalysis
          data={mockDemographicData}
          loading={false}
          error={null}
        />
      </div>

      {/* Financial Metrics Section */}
      <FinancialMetrics
        data={mockFinancialData}
        loading={false}
        error={null}
      />

      {/* Footer with demo info */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm text-gray-500">
          <div>
            Demo Mode - Using Mock Data for UI Testing
          </div>
          <div className="flex items-center space-x-4">
            <span className="flex items-center">
              <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
              Dashboard Components Active
            </span>
            <span className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
              UI Rendering Complete
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardDemo;