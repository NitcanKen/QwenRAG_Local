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
import { ChurnByTenure } from '../../types/dashboard';

interface ChurnByTenureChartProps {
  data: ChurnByTenure | null;
  loading?: boolean;
  error?: string | null;
}

const ChurnByTenureChart: React.FC<ChurnByTenureChartProps> = ({ 
  data, 
  loading, 
  error 
}) => {
  if (loading) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Churn Rate by Tenure
        </h3>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center animate-pulse">
          <div className="text-gray-400">Loading chart data...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card border-red-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Churn Rate by Tenure
        </h3>
        <div className="h-64 bg-red-50 rounded-lg flex items-center justify-center">
          <div className="text-red-600 text-sm">
            Error loading chart: {error}
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Churn Rate by Tenure
        </h3>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-gray-500">No data available</div>
        </div>
      </div>
    );
  }

  // Transform the data for the chart
  const chartData = Object.entries(data).map(([tenureGroup, stats]) => ({
    name: tenureGroup,
    customers: stats.customer_count,
    churned: stats.churned_count,
    churn_rate: Math.round(stats.churn_rate * 100),
    retained: stats.customer_count - stats.churned_count,
  }));

  // Sort by tenure group for logical ordering
  const sortedData = chartData.sort((a, b) => {
    const order = ['0-12 months', '13-24 months', '25-48 months', '49-72 months', '73+ months'];
    return order.indexOf(a.name) - order.indexOf(b.name);
  });

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Churn Rate by Tenure
        </h3>
        <div className="text-sm text-gray-500">
          Total: {sortedData.reduce((sum, item) => sum + item.customers, 0).toLocaleString()} customers
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={300}>
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
            stroke="#6b7280"
            fontSize={12}
            tick={{ fill: '#6b7280' }}
            label={{ 
              value: 'Churn Rate (%)', 
              angle: -90, 
              position: 'insideLeft',
              style: { textAnchor: 'middle', fill: '#6b7280' }
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
              if (name === 'churn_rate') return [`${value}%`, 'Churn Rate'];
              if (name === 'customers') return [value.toLocaleString(), 'Total Customers'];
              if (name === 'churned') return [value.toLocaleString(), 'Churned'];
              if (name === 'retained') return [value.toLocaleString(), 'Retained'];
              return [value, name];
            }}
            labelFormatter={(label) => `Tenure: ${label}`}
          />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }}
          />
          <Bar 
            dataKey="churn_rate" 
            fill="#ef4444" 
            name="Churn Rate (%)"
            radius={[2, 2, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
      
      {/* Summary statistics */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Highest Churn:</span>
            <span className="ml-2 font-medium text-red-600">
              {sortedData.reduce((max, item) => 
                item.churn_rate > max.churn_rate ? item : max
              ).name} ({sortedData.reduce((max, item) => 
                item.churn_rate > max.churn_rate ? item : max
              ).churn_rate}%)
            </span>
          </div>
          <div>
            <span className="text-gray-600">Lowest Churn:</span>
            <span className="ml-2 font-medium text-green-600">
              {sortedData.reduce((min, item) => 
                item.churn_rate < min.churn_rate ? item : min
              ).name} ({sortedData.reduce((min, item) => 
                item.churn_rate < min.churn_rate ? item : min
              ).churn_rate}%)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChurnByTenureChart;