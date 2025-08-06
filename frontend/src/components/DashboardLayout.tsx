import React, { useState } from 'react'
import { useAuth } from '../context/AuthContext.tsx'
import Sidebar from './Sidebar/Sidebar'
import Dashboard from './Dashboard/Dashboard'
import Email from './Email/Email'
import SharedFolder from './SharedFolder/SharedFolder'
import ChatInterface from '../pages/ChatInterface'
import { useAppContext } from '../context/AppContext'
 
const DashboardLayout: React.FC = () => {
  const [activeSection, setActiveSection] = useState('dashboard');
  const { currentUser, logout } = useAuth();
  const { isConnected } = useAppContext();
 
  const renderContent = () => {
    switch (activeSection) {
      case 'dashboard':
        return <Dashboard />;
      case 'email':
        return <Email />;
      case 'shared-folder':
        return <SharedFolder />;
      case 'chatbot':
        return (
          <div className="h-full">
            <ChatInterface />
          </div>
        );
      default:
        return <Dashboard />;
    }
  };
 
  // If chat is selected, render the full chat interface with original layout style
  if (activeSection === 'chatbot') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
        {/* Header similar to original Layout */}
        <header className="bg-white/80 backdrop-blur-md shadow-lg border-b border-blue-100/50">
          <div className="px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              {/* Logo */}
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl flex items-center justify-center shadow-lg">
                    <span className="text-white text-xl font-bold">B</span>
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-blue-700 bg-clip-text text-transparent">
                      BroadAxis-AI
                    </h1>
                    <p className="text-xs text-blue-600 -mt-1">RFP Management Platform</p>
                  </div>
                </div>
 
                {/* Live Status Indicator */}
                <div className={`flex items-center space-x-2 px-3 py-2 rounded-full text-xs font-medium shadow-lg backdrop-blur-md ${
                  isConnected
                    ? 'bg-green-500/20 border border-green-400/30'
                    : 'bg-red-500/20 border border-red-400/30'
                }`}>
                  <div className={`w-2 h-2 rounded-full ${
                    isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
                  }`}></div>
                  <span className={isConnected ? 'text-green-700' : 'text-red-700'}>
                    {isConnected ? 'Live' : 'Offline'}
                  </span>
                </div>
              </div>
 
              {/* Back to Dashboard Button */}
              <nav className="flex space-x-2">
                <button
                  onClick={() => setActiveSection('dashboard')}
                  className="px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 flex items-center space-x-2 text-blue-600 hover:text-blue-800 hover:bg-blue-100/60 hover:shadow-md"
                >
                  <span className="text-lg">←</span>
                  <span>Back to Dashboard</span>
                </button>
                <div className="px-4 py-2.5 rounded-xl text-sm font-medium bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg flex items-center space-x-2">
                  <span className="text-lg">💬</span>
                  <span>Chat</span>
                </div>
              </nav>
            </div>
          </div>
        </header>
 
        {/* Chat Interface */}
        <main>
          <ChatInterface />
        </main>
      </div>
    );
  }
 
  // For other sections, render with sidebar
  return (
    <div className="flex h-screen bg-gradient-to-br from-blue-50 to-white">
      <Sidebar
        activeSection={activeSection}
        setActiveSection={setActiveSection}
        onLogout={logout}
        currentUser={currentUser}
      />
      <main className="flex-1 overflow-auto">
        {renderContent()}
      </main>
    </div>
  );
};
 
export default DashboardLayout;