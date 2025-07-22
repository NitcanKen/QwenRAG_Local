import React from 'react';
import { MessageCircle, Upload, FileText, BarChart3 } from 'lucide-react';

const Chat: React.FC = () => {
  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          AI Chat Assistant
        </h1>
        <p className="text-gray-600 mt-2">
          Ask questions about customer analytics and uploaded documents
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chat Interface */}
        <div className="lg:col-span-2">
          <div className="card h-[600px] flex flex-col">
            <div className="flex items-center justify-between pb-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">
                Chat with DeepSeek-R1
              </h3>
              <span className="badge-success">
                <MessageCircle className="w-3 h-3 mr-1" />
                Online
              </span>
            </div>

            {/* Chat Messages Area */}
            <div className="flex-1 py-4 overflow-y-auto">
              <div className="space-y-4">
                {/* Example messages */}
                <div className="flex justify-start">
                  <div className="max-w-xs lg:max-w-md">
                    <div className="bg-gray-100 text-gray-800 p-3 rounded-lg">
                      <p className="text-sm">
                        Hello! I'm your AI assistant for customer churn analysis. 
                        I can help you understand your data, interpret analytics, 
                        and provide insights from uploaded documents.
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        AI Assistant • Just now
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end">
                  <div className="max-w-xs lg:max-w-md">
                    <div className="bg-primary-600 text-white p-3 rounded-lg">
                      <p className="text-sm">
                        What are the main reasons customers are churning?
                      </p>
                      <p className="text-xs text-primary-200 mt-2">
                        You • 2 minutes ago
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex justify-start">
                  <div className="max-w-xs lg:max-w-md">
                    <div className="bg-gray-100 text-gray-800 p-3 rounded-lg">
                      <p className="text-sm">
                        Based on the current analytics data, the main churn factors are:
                        <br />
                        1. High monthly charges (45.3% churn rate for electronic check users)
                        <br />
                        2. Short tenure (47.4% churn in first 12 months)
                        <br />
                        3. Month-to-month contracts
                        <br />
                        4. Fiber optic service customers
                      </p>
                      <div className="flex items-center mt-3 pt-2 border-t border-gray-200">
                        <BarChart3 className="w-4 h-4 text-gray-500 mr-2" />
                        <span className="text-xs text-gray-500">
                          Source: Dashboard Analytics
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        AI Assistant • 2 minutes ago
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Typing indicator placeholder */}
              <div className="flex justify-center mt-4">
                <div className="bg-gray-100 px-4 py-2 rounded-full">
                  <p className="text-sm text-gray-500">
                    Chat interface will be implemented in Stage 5.4
                  </p>
                </div>
              </div>
            </div>

            {/* Message Input */}
            <div className="pt-4 border-t border-gray-200">
              <div className="flex items-center space-x-2">
                <input
                  type="text"
                  placeholder="Ask about churn analytics or upload documents..."
                  className="input-field flex-1"
                  disabled
                />
                <button className="btn-primary px-4 py-2" disabled>
                  Send
                </button>
              </div>
              <div className="flex items-center justify-between mt-2">
                <div className="flex items-center space-x-4 text-xs text-gray-500">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      className="mr-1 rounded border-gray-300"
                      checked
                      disabled
                    />
                    Include Dashboard Data
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      className="mr-1 rounded border-gray-300"
                      checked
                      disabled
                    />
                    Include Documents
                  </label>
                </div>
                <div className="text-xs text-gray-500">
                  Powered by DeepSeek-R1:8b
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Document Upload */}
          <div className="card">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              Upload Documents
            </h4>
            <div className="space-y-4">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-sm text-gray-600">
                  Drop files here or click to upload
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  PDF, DOCX, TXT up to 50MB
                </p>
              </div>
              <button className="btn-secondary w-full" disabled>
                Choose Files
              </button>
            </div>
          </div>

          {/* Uploaded Documents */}
          <div className="card">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              Knowledge Base
            </h4>
            <div className="space-y-3">
              {[
                { name: 'Telco Industry Report 2024', type: 'Industry Report', size: '2.4 MB' },
                { name: 'Customer Feedback Analysis', type: 'Customer Feedback', size: '1.8 MB' },
                { name: 'Market Research - Fixed Wireless', type: 'Market Research', size: '3.1 MB' },
              ].map((doc, index) => (
                <div key={index} className="flex items-center p-3 bg-gray-50 rounded-lg">
                  <FileText className="w-5 h-5 text-gray-400 mr-3" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {doc.name}
                    </p>
                    <div className="flex items-center text-xs text-gray-500">
                      <span>{doc.type}</span>
                      <span className="mx-1">•</span>
                      <span>{doc.size}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chat Settings */}
          <div className="card">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              Chat Settings
            </h4>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Response Style
                </label>
                <select className="select-field" disabled>
                  <option>Balanced</option>
                  <option>Detailed</option>
                  <option>Concise</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Context Sources
                </label>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      className="mr-2 rounded border-gray-300"
                      checked
                      disabled
                    />
                    <span className="text-sm text-gray-600">Analytics Dashboard</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      className="mr-2 rounded border-gray-300"
                      checked
                      disabled
                    />
                    <span className="text-sm text-gray-600">Uploaded Documents</span>
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              Quick Questions
            </h4>
            <div className="space-y-2">
              {[
                'What is our current churn rate?',
                'Which customers are at highest risk?',
                'How do our metrics compare to industry benchmarks?',
                'What retention strategies work best?',
              ].map((question, index) => (
                <button
                  key={index}
                  className="w-full text-left p-3 text-sm text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
                  disabled
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;