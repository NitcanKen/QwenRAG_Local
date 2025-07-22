import React from 'react';

const Analytics: React.FC = () => {
  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Advanced Analytics
        </h1>
        <p className="text-gray-600 mt-2">
          Deep dive into customer churn patterns and predictive insights
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Financial Impact */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Financial Impact Analysis
          </h3>
          <div className="space-y-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-gray-600">Total Revenue</span>
                <span className="text-xl font-bold text-gray-900">$16.06M</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-primary-600 h-2 rounded-full w-full"></div>
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-gray-600">Lost Revenue</span>
                <span className="text-xl font-bold text-danger-600">$2.86M</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-danger-500 h-2 rounded-full w-1/6"></div>
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-gray-600">Retained Revenue</span>
                <span className="text-xl font-bold text-secondary-600">$13.20M</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-secondary-500 h-2 rounded-full w-5/6"></div>
              </div>
            </div>
          </div>
        </div>

        {/* Predictive Model Performance */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            ML Model Performance
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-600">Accuracy</span>
              <span className="text-lg font-bold text-gray-900">81.2%</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-600">Precision</span>
              <span className="text-lg font-bold text-gray-900">79.8%</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-600">Recall</span>
              <span className="text-lg font-bold text-gray-900">65.4%</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-600">F1 Score</span>
              <span className="text-lg font-bold text-gray-900">71.9%</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Payment Method Analysis */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Churn by Payment Method
          </h3>
          <div className="space-y-3">
            {[
              { method: 'Electronic Check', rate: '45.3%', customers: 2365, color: 'bg-danger-500' },
              { method: 'Mailed Check', rate: '19.1%', customers: 1612, color: 'bg-warning-500' },
              { method: 'Bank Transfer', rate: '16.9%', customers: 1544, color: 'bg-primary-500' },
              { method: 'Credit Card', rate: '15.2%', customers: 1522, color: 'bg-secondary-500' },
            ].map((payment, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <div className={`w-3 h-3 ${payment.color} rounded-full mr-3`}></div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{payment.method}</p>
                    <p className="text-xs text-gray-500">{payment.customers.toLocaleString()} customers</p>
                  </div>
                </div>
                <span className="text-sm font-bold text-gray-900">{payment.rate}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Tenure Groups */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Churn by Tenure Groups
          </h3>
          <div className="space-y-3">
            {[
              { group: '0-12 months', rate: '47.4%', customers: 2343, color: 'bg-danger-500' },
              { group: '13-24 months', rate: '35.2%', customers: 1655, color: 'bg-warning-500' },
              { group: '25-48 months', rate: '24.8%', customers: 1472, color: 'bg-primary-500' },
              { group: '49-72 months', rate: '6.6%', customers: 1573, color: 'bg-secondary-500' },
            ].map((tenure, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <div className={`w-3 h-3 ${tenure.color} rounded-full mr-3`}></div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{tenure.group}</p>
                    <p className="text-xs text-gray-500">{tenure.customers.toLocaleString()} customers</p>
                  </div>
                </div>
                <span className="text-sm font-bold text-gray-900">{tenure.rate}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Feature Importance */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Top Churn Predictors
          </h3>
          <div className="space-y-3">
            {[
              { feature: 'Monthly Charges', importance: 0.342, color: 'bg-primary-500' },
              { feature: 'Total Charges', importance: 0.298, color: 'bg-secondary-500' },
              { feature: 'Tenure', importance: 0.187, color: 'bg-warning-500' },
              { feature: 'Contract Type', importance: 0.123, color: 'bg-danger-500' },
              { feature: 'Payment Method', importance: 0.087, color: 'bg-gray-500' },
            ].map((feature, index) => (
              <div key={index} className="space-y-1">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-600">{feature.feature}</span>
                  <span className="text-sm font-bold text-gray-900">
                    {(feature.importance * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`${feature.color} h-2 rounded-full transition-all duration-300`}
                    style={{ width: `${feature.importance * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Customer Risk Distribution
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-6 bg-danger-50 rounded-lg border border-danger-200">
            <div className="text-3xl font-bold text-danger-600">1,867</div>
            <div className="text-sm font-medium text-danger-800 mt-1">High Risk</div>
            <div className="text-xs text-danger-600 mt-2">26.5% of customers</div>
          </div>
          <div className="text-center p-6 bg-warning-50 rounded-lg border border-warning-200">
            <div className="text-3xl font-bold text-warning-600">2,341</div>
            <div className="text-sm font-medium text-warning-800 mt-1">Medium Risk</div>
            <div className="text-xs text-warning-600 mt-2">33.2% of customers</div>
          </div>
          <div className="text-center p-6 bg-secondary-50 rounded-lg border border-secondary-200">
            <div className="text-3xl font-bold text-secondary-600">2,835</div>
            <div className="text-sm font-medium text-secondary-800 mt-1">Low Risk</div>
            <div className="text-xs text-secondary-600 mt-2">40.3% of customers</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;