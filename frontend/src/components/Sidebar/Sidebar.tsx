import React from 'react';
 
interface MenuItem {
  id: string;
  icon: string;
  label: string;
}
 
interface User {
  name?: string;
  email?: string;
}
 
interface SidebarProps {
  activeSection: string;
  setActiveSection: (section: string) => void;
  onLogout: () => void;
  currentUser: User | null;
}
 
const Sidebar: React.FC<SidebarProps> = ({
  activeSection,
  setActiveSection,
  onLogout,
  currentUser
}) => {
  const baseItems: MenuItem[] = [
    { id: 'dashboard', icon: '📊', label: 'Dashboard' },
    { id: 'email', icon: '📧', label: 'Email' },
    { id: 'shared-folder', icon: '📁', label: 'Shared Folder' },
    { id: 'chatbot', icon: '💬', label: 'Chat' }
  ];
  const menuItems: MenuItem[] = currentUser?.email && ['tariq@broadaxis.com','rohith.ganapuram@broadaxis.com'].includes(currentUser.email)
    ? [...baseItems, { id: 'trading', icon: '📈', label: 'Broadaxis Trading Planner' }]
    : baseItems;
 
  return (
    <div className="w-64 bg-white shadow-xl border-r border-gray-200 flex flex-col h-screen">
      {/* Sidebar Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl flex items-center justify-center shadow-lg">
            <span className="text-white text-xl font-bold">B</span>
          </div>
          <div>
            <h2 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-blue-700 bg-clip-text text-transparent">
              RFP Manager
            </h2>
          </div>
        </div>
      </div>
 
      {/* Navigation Menu */}
      <nav className="flex-1 p-4 space-y-2">
        {menuItems.map((item) => (
          <button
            key={item.id}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-left transition-all duration-200 ${
              activeSection === item.id
                ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg transform scale-105'
                : 'text-gray-700 hover:bg-blue-50 hover:text-blue-700 hover:shadow-md'
            }`}
            onClick={() => setActiveSection(item.id)}
          >
            <span className="text-xl">{item.icon}</span>
            <span className="font-medium">{item.label}</span>
          </button>
        ))}
      </nav>
 
      {/* Sidebar Footer */}
      <div className="p-4 border-t border-gray-200 space-y-4">
        {/* User Info */}
        <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-xl">
          <div className="w-10 h-10 bg-gradient-to-r from-gray-400 to-gray-500 rounded-full flex items-center justify-center">
            <span className="text-white text-lg">👤</span>
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-medium text-gray-900 truncate">
              {currentUser?.name || 'User'}
            </div>
            <div className="text-sm text-gray-500 truncate">
              {currentUser?.email || 'user@example.com'}
            </div>
          </div>
        </div>
 
        {/* Logout Button */}
        <button
          onClick={onLogout}
          className="w-full flex items-center space-x-3 px-4 py-3 text-red-600 hover:bg-red-50 rounded-xl transition-all duration-200 hover:shadow-md"
        >
          <span className="text-xl">🚪</span>
          <span className="font-medium">Logout</span>
        </button>
      </div>
    </div>
  );
};
 
export default Sidebar;
 