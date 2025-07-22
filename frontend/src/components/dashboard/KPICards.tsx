import React from 'react';
import { ArrowUpIcon, ArrowDownIcon } from '@heroicons/react/24/outline';
import { ChurnOverview } from '../../types/dashboard';

interface KPICardsProps {
  data: ChurnOverview | null;
  loading?: boolean;
  error?: string | null;
}

interface KPICardProps {
  label: string;
  value: string;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  isInverted?: boolean; // true for metrics where decrease is good (like churn rate)
  loading?: boolean;
}

const KPICard: React.FC<KPICardProps> = ({
  label,
  value,
  change,
  trend = 'neutral',
  isInverted = false,
  loading = false,
}) => {
  const getTrendColor = () => {
    if (!change) return 'text-gray-500';
    
    if (trend === 'neutral') return 'text-gray-500';
    
    // For inverted metrics (like churn rate), down is good
    if (isInverted) {
      return trend === 'down' ? 'text-green-600' : 'text-red-600';
    }
    
    // For normal metrics, up is good
    return trend === 'up' ? 'text-green-600' : 'text-red-600';
  };

  const TrendIcon = trend === 'up' ? ArrowUpIcon : ArrowDownIcon;

  if (loading) {
    return (
      <div className="stat-card animate-pulse">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-3"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
          </div>
          <div className="h-4 bg-gray-200 rounded w-12"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="stat-card hover:shadow-lg transition-shadow duration-200">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600">{label}</p>
          <p className="text-3xl font-bold text-gray-900 mt-2">{value}</p>
        </div>
        {change && trend !== 'neutral' && (
          <div className={`flex items-center text-sm font-medium ${getTrendColor()}`}>
            <TrendIcon className="w-4 h-4 mr-1" />
            {change}
          </div>
        )}
      </div>
    </div>
  );
};

const KPICards: React.FC<KPICardsProps> = ({ data, loading, error }) => {
  if (error) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="stat-card border-red-200">
            <div className="text-center text-red-600 text-sm">
              Unable to load data
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Calculate derived metrics
  const totalCustomers = data?.total_customers || 0;
  const churnRate = data?.churn_rate || 0;
  const retentionRate = data?.retention_rate || 0;
  const churned = data?.churned_customers || 0;

  // Calculate revenue at risk (placeholder calculation)
  const avgRevenuePerCustomer = 65; // Average monthly charges
  const revenueAtRisk = (churned * avgRevenuePerCustomer).toLocaleString();

  const kpiData = [
    {
      label: 'Total Customers',
      value: totalCustomers.toLocaleString(),
      change: undefined, // We don't have historical data for trend
      trend: 'neutral' as const,
      isInverted: false,
    },
    {
      label: 'Churn Rate',
      value: `${(churnRate * 100).toFixed(1)}%`,
      change: undefined, // Would need historical data
      trend: 'neutral' as const,
      isInverted: true,
    },
    {
      label: 'Revenue at Risk',
      value: `$${revenueAtRisk}`,
      change: undefined,
      trend: 'neutral' as const,
      isInverted: true,
    },
    {
      label: 'Retention Rate',
      value: `${(retentionRate * 100).toFixed(1)}%`,
      change: undefined,
      trend: 'neutral' as const,
      isInverted: false,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {kpiData.map((kpi, index) => (
        <KPICard
          key={index}
          label={kpi.label}
          value={kpi.value}
          change={kpi.change}
          trend={kpi.trend}
          isInverted={kpi.isInverted}
          loading={loading}
        />
      ))}
    </div>
  );
};

export default KPICards;