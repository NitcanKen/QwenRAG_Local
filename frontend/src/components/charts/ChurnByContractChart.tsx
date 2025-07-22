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
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { ChurnByContract } from '../../types/dashboard';

interface ChurnByContractChartProps {
  data: ChurnByContract | null;
  loading?: boolean;
  error?: string | null;
  chartType?: 'bar' | 'pie';
}

const COLORS = ['#ef4444', '#f97316', '#eab308'];

const ChurnByContractChart: React.FC<ChurnByContractChartProps> = ({ 
  data, 
  loading, 
  error,
  chartType = 'bar'
}) => {
  if (loading) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Churn Rate by Contract Type
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
          Churn Rate by Contract Type
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
          Churn Rate by Contract Type
        </h3>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-gray-500">No data available</div>
        </div>
      </div>
    );
  }

  // Transform the data for the chart
  const chartData = Object.entries(data).map(([contractType, stats], index) => ({
    name: contractType,
    customers: stats.customer_count,
    churned: stats.churned_count,
    churn_rate: Math.round(stats.churn_rate * 100),
    retained: stats.customer_count - stats.churned_count,
    color: COLORS[index % COLORS.length],
  }));

  // Sort by churn rate for better visualization
  const sortedData = chartData.sort((a, b) => b.churn_rate - a.churn_rate);

  const BarChartComponent = () => (
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
          labelFormatter={(label) => `Contract: ${label}`}
        />
        <Legend />
        <Bar 
          dataKey="churn_rate" 
          name="Churn Rate (%)"
          radius={[2, 2, 0, 0]}
        >
          {sortedData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );

  const PieChartComponent = () => (
    <div className="flex items-center">
      <ResponsiveContainer width="60%" height={250}>
        <PieChart>
          <Pie
            data={sortedData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, churn_rate }) => `${name}: ${churn_rate}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="churn_rate"
          >
            {sortedData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value: any) => [`${value}%`, 'Churn Rate']}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="flex-1 space-y-3">
        {sortedData.map((item, index) => (
          <div key={item.name} className="flex items-center">
            <div 
              className="w-4 h-4 rounded-sm mr-3"
              style={{ backgroundColor: item.color }}
            />
            <div className="flex-1">
              <div className="text-sm font-medium text-gray-900">{item.name}</div>
              <div className="text-xs text-gray-600">
                {item.customers.toLocaleString()} customers
              </div>
            </div>
            <div className="text-sm font-bold text-gray-900">
              {item.churn_rate}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Churn Rate by Contract Type
        </h3>
        <div className="text-sm text-gray-500">
          Total: {sortedData.reduce((sum, item) => sum + item.customers, 0).toLocaleString()} customers
        </div>
      </div>
      
      {chartType === 'bar' ? <BarChartComponent /> : <PieChartComponent />}
      
      {/* Key insights */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Highest Risk:</span>
            <span className="ml-2 font-medium text-red-600">
              {sortedData[0]?.name} ({sortedData[0]?.churn_rate}%)
            </span>
          </div>
          <div>
            <span className="text-gray-600">Most Stable:</span>
            <span className="ml-2 font-medium text-green-600">
              {sortedData[sortedData.length - 1]?.name} ({sortedData[sortedData.length - 1]?.churn_rate}%)
            </span>
          </div>
        </div>
        <div className="mt-2 text-xs text-gray-500">
          💡 Month-to-month contracts typically show higher churn rates due to flexibility
        </div>
      </div>
    </div>
  );
};

export default ChurnByContractChart;