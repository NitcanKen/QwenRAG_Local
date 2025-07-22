import React from 'react';

const SimpleTest: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-blue-600 mb-4">
        Dashboard Components Test
      </h1>
      <p className="text-gray-600 mb-6">
        Testing if basic React components work without type imports
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Simple KPI Cards */}
        <div className="bg-white p-6 rounded-lg shadow border">
          <p className="text-sm font-medium text-gray-600">Total Customers</p>
          <p className="text-3xl font-bold text-gray-900 mt-2">7,043</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow border">
          <p className="text-sm font-medium text-gray-600">Churn Rate</p>
          <p className="text-3xl font-bold text-red-600 mt-2">26.5%</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow border">
          <p className="text-sm font-medium text-gray-600">Revenue at Risk</p>
          <p className="text-3xl font-bold text-orange-600 mt-2">$2.86M</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow border">
          <p className="text-sm font-medium text-gray-600">Retention Rate</p>
          <p className="text-3xl font-bold text-green-600 mt-2">73.5%</p>
        </div>
      </div>
      
      <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-900 mb-2">✅ Component Test Status</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• React rendering: Working</li>
          <li>• Tailwind CSS: Working</li>
          <li>• Component structure: Working</li>
          <li>• TypeScript: Working (basic)</li>
        </ul>
      </div>
      
      <div className="mt-6 bg-green-50 border border-green-200 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-green-900 mb-2">🎯 Next Steps</h3>
        <p className="text-sm text-green-800">
          This confirms the basic React setup is working. The dashboard components 
          are ready for integration once the type import issues are resolved.
        </p>
      </div>
    </div>
  );
};

export default SimpleTest;