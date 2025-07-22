import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  BarChart3, 
  TrendingUp, 
  MessageCircle, 
  Zap
} from 'lucide-react';

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<any>;
  description: string;
}

const navigation: NavItem[] = [
  {
    name: 'Dashboard',
    href: '/',
    icon: BarChart3,
    description: 'Overview and key metrics'
  },
  {
    name: 'Analytics',
    href: '/analytics',
    icon: TrendingUp,
    description: 'Deep dive analysis'
  },
  {
    name: 'AI Chat',
    href: '/chat',
    icon: MessageCircle,
    description: 'Ask questions with AI'
  },
  // {
  //   name: 'Customers',
  //   href: '/customers',
  //   icon: Users,
  //   description: 'Customer management'
  // },
  // {
  //   name: 'Documents',
  //   href: '/documents',
  //   icon: FileText,
  //   description: 'Knowledge base'
  // },
  // {
  //   name: 'Real-time',
  //   href: '/realtime',
  //   icon: Activity,
  //   description: 'Live updates'
  // },
];

const Sidebar: React.FC = () => {
  const location = useLocation();

  return (
    <div className="hidden md:flex md:flex-shrink-0">
      <div className="flex flex-col w-64">
        <div className="flex flex-col flex-grow bg-white border-r border-gray-200 overflow-y-auto">
          {/* Logo */}
          <div className="flex items-center justify-center h-16 px-4 bg-primary-600">
            <div className="flex items-center">
              <Zap className="h-8 w-8 text-white" />
              <div className="ml-3">
                <h1 className="text-lg font-bold text-white">ChurnScope</h1>
                <p className="text-xs text-primary-200">AI-Powered Analytics</p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex-1 px-3 py-6">
            <nav className="space-y-1">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href;
                const Icon = item.icon;

                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`nav-link ${isActive ? 'nav-link-active' : ''}`}
                  >
                    <Icon className="mr-3 h-5 w-5 flex-shrink-0" />
                    <div className="flex-1">
                      <div className="text-sm font-medium">{item.name}</div>
                      <div className="text-xs text-gray-500">{item.description}</div>
                    </div>
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* Status indicators */}
          <div className="px-3 pb-6">
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-900 mb-3">
                System Status
              </h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">API</span>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-secondary-500 rounded-full mr-2"></div>
                    <span className="text-xs text-secondary-600">Online</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">ML Model</span>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-secondary-500 rounded-full mr-2"></div>
                    <span className="text-xs text-secondary-600">Ready</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">RAG System</span>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-secondary-500 rounded-full mr-2"></div>
                    <span className="text-xs text-secondary-600">Active</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">DeepSeek AI</span>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-secondary-500 rounded-full mr-2"></div>
                    <span className="text-xs text-secondary-600">Available</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;