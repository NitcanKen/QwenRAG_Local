import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { ChurnByService } from '../../types/dashboard';

interface ServiceAnalysisProps {
  data: ChurnByService | null;
  loading?: boolean;
  error?: string | null;
}

const ServiceAnalysis: React.FC<ServiceAnalysisProps> = ({ data, loading, error }) => {
  if (loading) {
    return (
      <div className="card lg:col-span-2">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Churn by Internet Service
        </h3>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center animate-pulse">
          <div className="text-gray-400">Loading service analysis...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card lg:col-span-2 border-red-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Churn by Internet Service
        </h3>
        <div className="h-64 bg-red-50 rounded-lg flex items-center justify-center">
          <div className="text-red-600 text-sm">
            Error loading service analysis: {error}
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="card lg:col-span-2">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Churn by Internet Service
        </h3>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-gray-500">No service data available</div>
        </div>
      </div>
    );
  }

  // Transform the data for the chart
  const chartData = Object.entries(data).map(([serviceType, stats]) => ({
    name: serviceType.replace('internet_service_', '').replace('_', ' ').toUpperCase(),
    customers: stats.customer_count,
    churned: stats.churned_count,
    churn_rate: Math.round(stats.churn_rate * 100),
    retained: stats.customer_count - stats.churned_count,
    retention_rate: Math.round((1 - stats.churn_rate) * 100),
  }));

  // Sort by customer count for better visualization
  const sortedData = chartData.sort((a, b) => b.customers - a.customers);

  // Service colors
  const getServiceColor = (serviceName: string) => {
    switch (serviceName.toLowerCase()) {
      case 'fiber optic': return '#ef4444'; // Red for high churn
      case 'dsl': return '#f97316'; // Orange for medium
      case 'no': return '#22c55e'; // Green for no service
      default: return '#6b7280'; // Gray fallback
    }
  };

  return (
    <div className="card lg:col-span-2">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Churn Analysis by Internet Service
          </h3>
          <p className="text-sm text-gray-600 mt-1">
            Customer distribution and churn rates across internet service types
          </p>
        </div>
        <div className="text-sm text-gray-500">
          Total: {sortedData.reduce((sum, item) => sum + item.customers, 0).toLocaleString()} customers
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={sortedData}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="name" 
            stroke="#6b7280"
            fontSize={12}
            tick={{ fill: '#6b7280' }}
          />
          <YAxis 
            yAxisId="customers"
            orientation="left"
            stroke="#6b7280"
            fontSize={12}
            tick={{ fill: '#6b7280' }}
            label={{ 
              value: 'Customer Count', 
              angle: -90, 
              position: 'insideLeft',
              style: { textAnchor: 'middle', fill: '#6b7280' }
            }}
          />
          <YAxis 
            yAxisId="rate"
            orientation="right"
            stroke="#ef4444"
            fontSize={12}
            tick={{ fill: '#ef4444' }}
            label={{ 
              value: 'Churn Rate (%)', 
              angle: 90, 
              position: 'insideRight',
              style: { textAnchor: 'middle', fill: '#ef4444' }
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#fff',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            }}
            formatter={(value: any, name: string) => {
              if (name === 'customers') return [value.toLocaleString(), 'Total Customers'];
              if (name === 'churn_rate') return [`${value}%`, 'Churn Rate'];
              if (name === 'churned') return [value.toLocaleString(), 'Churned'];
              if (name === 'retained') return [value.toLocaleString(), 'Retained'];
              return [value, name];
            }}
            labelFormatter={(label) => `Service: ${label}`}
          />
          <Legend />
          <Bar 
            yAxisId="customers"
            dataKey="customers" 
            name="Total Customers"
            fill="#60a5fa"
            radius={[2, 2, 0, 0]}
          />
          <Bar 
            yAxisId="rate"
            dataKey="churn_rate" 
            name="Churn Rate (%)"
            fill="#ef4444"
            radius={[2, 2, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
      
      {/* Service breakdown cards */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h4 className="text-md font-medium text-gray-900 mb-4">Service Breakdown</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {sortedData.map((service) => (
            <div 
              key={service.name} 
              className="bg-gray-50 rounded-lg p-4 border-l-4"
              style={{ borderLeftColor: getServiceColor(service.name) }}
            >
              <div className="flex items-center justify-between mb-2">
                <h5 className="font-medium text-gray-900">{service.name}</h5>
                <span 
                  className="text-xs font-bold px-2 py-1 rounded"
                  style={{
                    backgroundColor: service.churn_rate > 40 ? '#fee2e2' : service.churn_rate > 25 ? '#fef3c7' : '#dcfce7',
                    color: service.churn_rate > 40 ? '#dc2626' : service.churn_rate > 25 ? '#d97706' : '#16a34a',
                  }}
                >
                  {service.churn_rate}% churn
                </span>
              </div>
              <div className="space-y-2 text-sm text-gray-600">
                <div className="flex justify-between">
                  <span>Total Customers:</span>
                  <span className="font-medium">{service.customers.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Churned:</span>
                  <span className="font-medium text-red-600">{service.churned.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Retained:</span>
                  <span className="font-medium text-green-600">{service.retained.toLocaleString()}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key insights */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="bg-blue-50 rounded-lg p-4">
          <h5 className="text-sm font-medium text-blue-900 mb-2">💡 Key Insights</h5>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Fiber optic customers typically show higher churn due to competitive alternatives</li>
            <li>• DSL customers may churn seeking faster internet speeds</li>
            <li>• Customers without internet service have different retention drivers</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ServiceAnalysis;