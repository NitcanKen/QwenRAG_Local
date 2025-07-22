import React from 'react';
import { DemographicAnalysis as DemographicData } from '../../types/dashboard';

interface DemographicAnalysisProps {
  data: DemographicData | null;
  loading?: boolean;
  error?: string | null;
}

interface DemographicItemProps {
  label: string;
  churnRate: number;
  customerCount: number;
  churnedCount: number;
  color: string;
}

const DemographicItem: React.FC<DemographicItemProps> = ({
  label,
  churnRate,
  customerCount,
  churnedCount,
  color,
}) => {
  const percentage = Math.round(churnRate * 100);
  
  return (
    <div className="flex items-center justify-between py-3 border-b border-gray-100 last:border-b-0">
      <div className="flex items-center">
        <div 
          className="w-3 h-3 rounded-full mr-3 flex-shrink-0"
          style={{ backgroundColor: color }}
        />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-gray-900 truncate">
            {label}
          </div>
          <div className="text-xs text-gray-500">
            {customerCount.toLocaleString()} customers
          </div>
        </div>
      </div>
      <div className="text-right ml-4">
        <div className="text-sm font-bold text-gray-900">
          {percentage}%
        </div>
        <div className="text-xs text-gray-500">
          {churnedCount.toLocaleString()} churned
        </div>
      </div>
    </div>
  );
};

interface DemographicSectionProps {
  title: string;
  data: Record<string, { customer_count: number; churned_count: number; churn_rate: number }>;
  colors: string[];
  icon?: string;
  loading?: boolean;
}

const DemographicSection: React.FC<DemographicSectionProps> = ({
  title,
  data,
  colors,
  icon,
  loading,
}) => {
  if (loading) {
    return (
      <div className="space-y-3">
        <div className="flex items-center mb-3">
          {icon && <span className="mr-2">{icon}</span>}
          <h4 className="text-md font-medium text-gray-900">{title}</h4>
        </div>
        {[1, 2, 3].map((i) => (
          <div key={i} className="animate-pulse">
            <div className="flex items-center justify-between py-3">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-gray-200 rounded-full mr-3" />
                <div className="space-y-1">
                  <div className="h-4 bg-gray-200 rounded w-20" />
                  <div className="h-3 bg-gray-200 rounded w-16" />
                </div>
              </div>
              <div className="text-right space-y-1">
                <div className="h-4 bg-gray-200 rounded w-10" />
                <div className="h-3 bg-gray-200 rounded w-12" />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  const sortedData = Object.entries(data).sort((a, b) => b[1].churn_rate - a[1].churn_rate);

  return (
    <div>
      <div className="flex items-center mb-4">
        {icon && <span className="mr-2 text-lg">{icon}</span>}
        <h4 className="text-md font-medium text-gray-900">{title}</h4>
      </div>
      <div className="space-y-1">
        {sortedData.map(([key, stats], index) => (
          <DemographicItem
            key={key}
            label={key}
            churnRate={stats.churn_rate}
            customerCount={stats.customer_count}
            churnedCount={stats.churned_count}
            color={colors[index % colors.length]}
          />
        ))}
      </div>
    </div>
  );
};

const DemographicAnalysis: React.FC<DemographicAnalysisProps> = ({ 
  data, 
  loading, 
  error 
}) => {
  if (loading) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Customer Demographics
        </h3>
        <div className="space-y-6">
          <DemographicSection
            title="Gender Distribution"
            data={{}}
            colors={[]}
            loading={true}
          />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card border-red-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Customer Demographics
        </h3>
        <div className="h-40 bg-red-50 rounded-lg flex items-center justify-center">
          <div className="text-red-600 text-sm">
            Error loading demographics: {error}
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Customer Demographics
        </h3>
        <div className="h-40 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-gray-500">No demographic data available</div>
        </div>
      </div>
    );
  }

  const genderColors = ['#3b82f6', '#ec4899']; // Blue, Pink
  const seniorColors = ['#f59e0b', '#10b981']; // Amber, Emerald
  const familyColors = ['#8b5cf6', '#06b6d4', '#84cc16']; // Purple, Cyan, Lime

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-6">
        Customer Demographics
      </h3>
      
      <div className="space-y-8">
        {/* Gender Analysis */}
        {data.gender_analysis && (
          <DemographicSection
            title="Gender Distribution"
            data={data.gender_analysis}
            colors={genderColors}
            icon="👥"
          />
        )}

        {/* Senior Citizen Analysis */}
        {data.senior_citizen_analysis && (
          <DemographicSection
            title="Age Demographics"
            data={data.senior_citizen_analysis}
            colors={seniorColors}
            icon="👴"
          />
        )}

        {/* Family Analysis */}
        {data.family_analysis && (
          <DemographicSection
            title="Family Status"
            data={data.family_analysis}
            colors={familyColors}
            icon="👨‍👩‍👧‍👦"
          />
        )}
      </div>

      {/* Summary insights */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4">
          <h5 className="text-sm font-medium text-indigo-900 mb-3">
            🎯 Demographic Insights
          </h5>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-indigo-800">
            {data.gender_analysis && (
              <div>
                <strong>Gender Impact:</strong>
                <div className="text-xs mt-1">
                  {Object.entries(data.gender_analysis).sort((a, b) => b[1].churn_rate - a[1].churn_rate)[0][1].churn_rate > 
                   Object.entries(data.gender_analysis).sort((a, b) => a[1].churn_rate - b[1].churn_rate)[0][1].churn_rate
                    ? 'Gender shows variation in churn patterns'
                    : 'Gender shows similar churn patterns'
                  }
                </div>
              </div>
            )}
            
            {data.senior_citizen_analysis && (
              <div>
                <strong>Age Factor:</strong>
                <div className="text-xs mt-1">
                  Senior citizens may have different service preferences
                </div>
              </div>
            )}
            
            {data.family_analysis && (
              <div>
                <strong>Family Impact:</strong>
                <div className="text-xs mt-1">
                  Family status influences retention rates
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DemographicAnalysis;