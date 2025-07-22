import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../services/api';

// Import new components
import KPICards from '../components/dashboard/KPICards';
import ChurnByTenureChart from '../components/charts/ChurnByTenureChart';
import ChurnByContractChart from '../components/charts/ChurnByContractChart';
import ServiceAnalysis from '../components/dashboard/ServiceAnalysis';
import DemographicAnalysis from '../components/dashboard/DemographicAnalysis';
import FinancialMetrics from '../components/dashboard/FinancialMetrics';

const Dashboard: React.FC = () => {
  // Fetch dashboard data using React Query
  const {
    data: churnOverview,
    isLoading: overviewLoading,
    error: overviewError,
  } = useQuery({
    queryKey: ['churnOverview'],
    queryFn: () => apiClient.getChurnOverview(),
    refetchInterval: 300000, // Refetch every 5 minutes
    staleTime: 60000, // Consider data stale after 1 minute
  });

  const {
    data: tenureData,
    isLoading: tenureLoading,
    error: tenureError,
  } = useQuery({
    queryKey: ['churnByTenure'],
    queryFn: () => apiClient.getChurnByTenure(),
    refetchInterval: 300000,
    staleTime: 60000,
  });

  const {
    data: contractData,
    isLoading: contractLoading,
    error: contractError,
  } = useQuery({
    queryKey: ['churnByContract'],
    queryFn: () => apiClient.getChurnByContract(),
    refetchInterval: 300000,
    staleTime: 60000,
  });

  const {
    data: serviceData,
    isLoading: serviceLoading,
    error: serviceError,
  } = useQuery({
    queryKey: ['churnByService'],
    queryFn: () => apiClient.getChurnByService(),
    refetchInterval: 300000,
    staleTime: 60000,
  });

  const {
    data: demographicData,
    isLoading: demographicLoading,
    error: demographicError,
  } = useQuery({
    queryKey: ['demographicAnalysis'],
    queryFn: () => apiClient.getDemographicAnalysis(),
    refetchInterval: 300000,
    staleTime: 60000,
  });

  const {
    data: financialData,
    isLoading: financialLoading,
    error: financialError,
  } = useQuery({
    queryKey: ['financialMetrics'],
    queryFn: () => apiClient.getFinancialMetrics(),
    refetchInterval: 300000,
    staleTime: 60000,
  });

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header Section */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Customer Churn Dashboard
            </h1>
            <p className="text-gray-600 mt-2">
              Real-time analytics and insights into customer churn patterns
            </p>
          </div>
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
              Live Data
            </div>
            <span>•</span>
            <span>Updated every 5 minutes</span>
          </div>
        </div>
      </div>

      {/* KPI Cards Section */}
      <KPICards 
        data={churnOverview} 
        loading={overviewLoading} 
        error={overviewError?.message} 
      />

      {/* Main Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <ChurnByTenureChart
          data={tenureData}
          loading={tenureLoading}
          error={tenureError?.message}
        />
        <ChurnByContractChart
          data={contractData}
          loading={contractLoading}
          error={contractError?.message}
        />
      </div>

      {/* Service Analysis and Demographics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <ServiceAnalysis
          data={serviceData}
          loading={serviceLoading}
          error={serviceError?.message}
        />
        <DemographicAnalysis
          data={demographicData}
          loading={demographicLoading}
          error={demographicError?.message}
        />
      </div>

      {/* Financial Metrics Section */}
      <FinancialMetrics
        data={financialData}
        loading={financialLoading}
        error={financialError?.message}
      />

      {/* Footer with last update info */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm text-gray-500">
          <div>
            {churnOverview?.last_updated && (
              <span>Last updated: {new Date(churnOverview.last_updated).toLocaleString()}</span>
            )}
          </div>
          <div className="flex items-center space-x-4">
            <span className="flex items-center">
              <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
              Analytics Engine Active
            </span>
            <span className="flex items-center">
              <div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>
              ML Predictions Ready
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;