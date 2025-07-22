import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from 'recharts';
import { FinancialMetrics as FinancialData } from '../../types/dashboard';

interface FinancialMetricsProps {
  data: FinancialData | null;
  loading?: boolean;
  error?: string | null;
}

interface RevenueCardProps {
  title: string;
  amount: number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  color: string;
  loading?: boolean;
}

const RevenueCard: React.FC<RevenueCardProps> = ({
  title,
  amount,
  subtitle,
  trend = 'neutral',
  color,
  loading,
}) => {
  if (loading) {
    return (
      <div className="bg-white rounded-lg p-6 shadow-sm border animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-3/4 mb-3"></div>
        <div className="h-8 bg-gray-200 rounded w-1/2 mb-2"></div>
        <div className="h-3 bg-gray-200 rounded w-2/3"></div>
      </div>
    );
  }

  const formatCurrency = (value: number) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(2)}M`;
    } else if (value >= 1000) {
      return `$${(value / 1000).toFixed(0)}K`;
    }
    return `$${value.toFixed(2)}`;
  };

  return (
    <div className="bg-white rounded-lg p-6 shadow-sm border hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-medium text-gray-600">{title}</h4>
        <div 
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
      <div className="text-2xl font-bold text-gray-900 mb-2">
        {formatCurrency(amount)}
      </div>
      {subtitle && (
        <p className="text-sm text-gray-500">{subtitle}</p>
      )}
    </div>
  );
};

const FinancialMetrics: React.FC<FinancialMetricsProps> = ({ 
  data, 
  loading, 
  error 
}) => {
  if (loading) {
    return (
      <div className="space-y-6">
        {/* Revenue Impact Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <RevenueCard
              key={i}
              title=""
              amount={0}
              color=""
              loading={true}
            />
          ))}
        </div>
        
        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Revenue Distribution by Charges
            </h3>
            <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center animate-pulse">
              <div className="text-gray-400">Loading financial data...</div>
            </div>
          </div>
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Churn Impact by Charge Group
            </h3>
            <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center animate-pulse">
              <div className="text-gray-400">Loading churn impact...</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div className="card border-red-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Financial Metrics
          </h3>
          <div className="h-32 bg-red-50 rounded-lg flex items-center justify-center">
            <div className="text-red-600 text-sm">
              Error loading financial data: {error}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="space-y-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Financial Metrics
          </h3>
          <div className="h-32 bg-gray-50 rounded-lg flex items-center justify-center">
            <div className="text-gray-500">No financial data available</div>
          </div>
        </div>
      </div>
    );
  }

  const { revenue_impact, charges_distribution } = data;

  // Transform charges distribution data for charts
  const chargesChartData = Object.entries(charges_distribution).map(([group, stats]) => ({
    name: group.replace('_', '-').replace('charges', ''),
    customers: stats.customer_count,
    churned: stats.churned_count,
    retained: stats.customer_count - stats.churned_count,
    churn_rate: Math.round(stats.churn_rate * 100),
    avg_monthly: stats.avg_monthly_charges,
    avg_total: stats.avg_total_charges,
    revenue: stats.customer_count * stats.avg_monthly_charges,
  }));

  // Sort by monthly charges for logical ordering
  const sortedChargesData = chargesChartData.sort((a, b) => a.avg_monthly - b.avg_monthly);

  return (
    <div className="space-y-6">
      {/* Revenue Impact Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <RevenueCard
          title="Total Revenue"
          amount={revenue_impact.total_revenue}
          subtitle="All customers combined"
          color="#10b981"
        />
        <RevenueCard
          title="Retained Revenue"
          amount={revenue_impact.retained_revenue}
          subtitle="From loyal customers"
          color="#3b82f6"
        />
        <RevenueCard
          title="Lost Revenue"
          amount={revenue_impact.lost_revenue}
          subtitle="From churned customers"
          color="#ef4444"
        />
        <RevenueCard
          title="Avg Revenue/User"
          amount={revenue_impact.avg_revenue_per_user}
          subtitle="Monthly per customer"
          color="#8b5cf6"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Revenue Distribution Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Revenue Distribution by Charge Group
            </h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={sortedChargesData}>
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
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
              />
              <Tooltip
                formatter={(value: any, name: string) => {
                  if (name === 'revenue') {
                    return [`$${(value / 1000).toFixed(0)}K`, 'Monthly Revenue'];
                  }
                  return [value, name];
                }}
                labelFormatter={(label) => `Charge Group: ${label}`}
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                }}
              />
              <Area
                type="monotone"
                dataKey="revenue"
                stroke="#10b981"
                fill="#10b981"
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Churn Impact by Charge Group */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Churn Impact by Charge Group
            </h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={sortedChargesData}>
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
                formatter={(value: any, name: string) => {
                  if (name === 'churn_rate') return [`${value}%`, 'Churn Rate'];
                  if (name === 'avg_monthly') return [`$${value.toFixed(2)}`, 'Avg Monthly'];
                  return [value.toLocaleString(), name];
                }}
                labelFormatter={(label) => `Charge Group: ${label}`}
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                }}
              />
              <Legend />
              <Bar dataKey="churn_rate" fill="#ef4444" name="Churn Rate (%)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Financial Breakdown */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-6">
          Financial Impact by Customer Segments
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-medium text-gray-900">Charge Group</th>
                <th className="text-right py-3 px-4 font-medium text-gray-900">Customers</th>
                <th className="text-right py-3 px-4 font-medium text-gray-900">Churn Rate</th>
                <th className="text-right py-3 px-4 font-medium text-gray-900">Avg Monthly</th>
                <th className="text-right py-3 px-4 font-medium text-gray-900">Total Revenue</th>
                <th className="text-right py-3 px-4 font-medium text-gray-900">Revenue at Risk</th>
              </tr>
            </thead>
            <tbody>
              {sortedChargesData.map((group) => (
                <tr key={group.name} className="border-b border-gray-100">
                  <td className="py-3 px-4 font-medium text-gray-900">
                    {group.name.charAt(0).toUpperCase() + group.name.slice(1)}
                  </td>
                  <td className="text-right py-3 px-4 text-gray-600">
                    {group.customers.toLocaleString()}
                  </td>
                  <td className="text-right py-3 px-4">
                    <span className={`font-medium ${
                      group.churn_rate > 30 ? 'text-red-600' : 
                      group.churn_rate > 20 ? 'text-yellow-600' : 'text-green-600'
                    }`}>
                      {group.churn_rate}%
                    </span>
                  </td>
                  <td className="text-right py-3 px-4 text-gray-600">
                    ${group.avg_monthly.toFixed(2)}
                  </td>
                  <td className="text-right py-3 px-4 font-medium text-gray-900">
                    ${(group.revenue / 1000).toFixed(0)}K
                  </td>
                  <td className="text-right py-3 px-4 font-medium text-red-600">
                    ${((group.churned * group.avg_monthly) / 1000).toFixed(0)}K
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Financial Insights */}
      <div className="card bg-gradient-to-r from-green-50 to-emerald-50">
        <h4 className="text-lg font-semibold text-green-900 mb-4">
          💰 Financial Insights & Recommendations
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-green-800">
          <div>
            <h5 className="font-medium mb-2">Revenue Protection</h5>
            <ul className="space-y-1 text-green-700">
              <li>• Focus retention efforts on high-value customers</li>
              <li>• Monitor customers with declining payment behavior</li>
              <li>• Implement targeted offers for at-risk segments</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium mb-2">Growth Opportunities</h5>
            <ul className="space-y-1 text-green-700">
              <li>• Upsell services to low-charge, loyal customers</li>
              <li>• Analyze competitors pricing in high-churn segments</li>
              <li>• Consider value-based pricing strategies</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FinancialMetrics;