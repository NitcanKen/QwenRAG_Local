import React from 'react';
import { Bell, Search, Settings, User, RefreshCw } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-white border-b border-gray-200">
      <div className="flex items-center justify-between px-6 py-4">
        {/* Search */}
        <div className="flex-1 max-w-lg">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg leading-5 bg-gray-50 placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-primary-500 focus:border-primary-500 focus:bg-white"
              placeholder="Search customers, metrics, or ask a question..."
              disabled
            />
          </div>
        </div>

        {/* Right side actions */}
        <div className="flex items-center space-x-4">
          {/* Data refresh indicator */}
          <div className="flex items-center text-sm text-gray-500">
            <RefreshCw className="h-4 w-4 mr-2" />
            <span>Last updated: 2 min ago</span>
          </div>

          {/* Quick stats */}
          <div className="hidden lg:flex items-center space-x-6 px-6 py-2 bg-gray-50 rounded-lg">
            <div className="text-center">
              <div className="text-lg font-bold text-gray-900">7,043</div>
              <div className="text-xs text-gray-500">Customers</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-danger-600">26.5%</div>
              <div className="text-xs text-gray-500">Churn Rate</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-secondary-600">81.2%</div>
              <div className="text-xs text-gray-500">ML Accuracy</div>
            </div>
          </div>

          {/* Notifications */}
          <button className="relative p-2 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-lg">
            <Bell className="h-6 w-6" />
            <span className="absolute top-1 right-1 block h-2 w-2 bg-danger-500 rounded-full"></span>
          </button>

          {/* Settings */}
          <button className="p-2 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-lg">
            <Settings className="h-6 w-6" />
          </button>

          {/* User menu */}
          <div className="relative">
            <button className="flex items-center text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500">
              <div className="h-8 w-8 bg-primary-600 rounded-full flex items-center justify-center">
                <User className="h-5 w-5 text-white" />
              </div>
              <div className="ml-3 text-left">
                <div className="text-sm font-medium text-gray-900">Data Analyst</div>
                <div className="text-xs text-gray-500">analyst@company.com</div>
              </div>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;