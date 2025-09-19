import React, { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import { apiClient } from '../utils/api'
import { AppSettings } from '../types'
import { useAuth } from '../context/AuthContext'

const Settings: React.FC = () => {
  const { currentUser } = useAuth()
  const [settings, setSettings] = useState<AppSettings>({
    model: 'claude-3-7-sonnet-20250219',
    enabledTools: [],
    autoAnalyze: false,
    theme: 'light'
  })
  const [serverStatus, setServerStatus] = useState<string>('checking')
  
  // Generate user profile from auth context
  const userProfile = {
    name: currentUser?.name || 'User',
    email: currentUser?.email || 'user@example.com',
    role: 'Broadaxis-AI',
    department: 'Business Development',
    joinDate: '2024-01-01', // Default join date
    avatar: (currentUser?.name || 'User').split(' ').map(n => n[0]).join('').slice(0, 2)
  }
  const [paymentInfo] = useState({
    plan: 'Professional',
    status: 'Active',
    nextBilling: '2025-02-15',
    usage: {
      tokens: 125000,
      limit: 500000,
      percentage: 25
    }
  })

  useEffect(() => {
    loadSettings()
  }, [])

  const loadSettings = async () => {
    const savedSettings = localStorage.getItem('broadaxis-settings')
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings))
    }
    
    // Initialize MCP server and check status
    try {
      setServerStatus('checking')
      const result = await apiClient.initializeMCP()
      setServerStatus(result.status === 'success' ? 'connected' : 'error')
    } catch (error) {
      console.error('Failed to initialize MCP:', error)
      setServerStatus('error')
    }
  }

  const handleModelChange = (model: string) => {
    setSettings(prev => ({ ...prev, model }))
  }

  const handleThemeChange = (theme: 'light' | 'dark') => {
    setSettings(prev => ({ ...prev, theme }))
    // Apply theme immediately
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
      document.body.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
      document.body.classList.remove('dark')
    }
  }

  const handleAutoAnalyzeChange = (autoAnalyze: boolean) => {
    setSettings(prev => ({ ...prev, autoAnalyze }))
  }

  const saveSettings = () => {
    localStorage.setItem('broadaxis-settings', JSON.stringify(settings))
    toast.success('Settings saved successfully')
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-green-600 bg-green-100'
      case 'error': return 'text-red-600 bg-red-100'
      default: return 'text-yellow-600 bg-yellow-100'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return '🟢'
      case 'error': return '🔴'
      default: return '🟡'
    }
  }

  return (
    <div className={`w-full p-6 min-h-screen transition-colors duration-300 ${
      settings.theme === 'dark' 
        ? 'bg-gradient-to-br from-gray-900 to-gray-800' 
        : 'bg-gradient-to-br from-blue-50 to-white'
    }`}>
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 flex justify-between items-start">
          <div>
            <h1 className={`text-2xl font-bold mb-2 transition-colors duration-300 ${
              settings.theme === 'dark' ? 'text-white' : 'text-blue-900'
            }`}>⚙️ Settings</h1>
            <p className={`text-sm transition-colors duration-300 ${
              settings.theme === 'dark' ? 'text-gray-300' : 'text-blue-600'
            }`}>Manage your account, preferences, and subscription</p>
          </div>
          
          {/* Save Settings Button - Top Right */}
          <button
            onClick={saveSettings}
            className={`px-5 py-2 rounded-lg transition-all duration-200 shadow-md hover:shadow-lg font-medium text-xs flex items-center ${
              settings.theme === 'dark'
                ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800'
                : 'bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800'
            }`}
          >
            <span className="mr-2">💾</span>
            Save Settings
          </button>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Left Column - User Profile */}
          <div className="xl:col-span-2 space-y-6">
            {/* User Profile Card */}
            <div className={`rounded-2xl shadow-lg p-6 border transition-colors duration-300 ${
              settings.theme === 'dark' 
                ? 'bg-gray-800 border-gray-700' 
                : 'bg-white border-blue-100'
            }`}>
              <h2 className={`text-lg font-bold mb-4 flex items-center transition-colors duration-300 ${
                settings.theme === 'dark' ? 'text-white' : 'text-blue-900'
              }`}>
                <span className="mr-2">👤</span>
                User Profile
              </h2>
              
              <div className="flex items-start space-x-6">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                  <span className="text-lg font-bold text-white">{userProfile.avatar}</span>
                </div>
                
                <div className="flex-1">
                  <h3 className={`text-lg font-bold mb-1 transition-colors duration-300 ${
                    settings.theme === 'dark' ? 'text-white' : 'text-blue-900'
                  }`}>{userProfile.name}</h3>
                  <p className={`text-sm mb-3 transition-colors duration-300 ${
                    settings.theme === 'dark' ? 'text-gray-300' : 'text-blue-600'
                  }`}>{userProfile.role}</p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className={`rounded-xl p-4 transition-colors duration-300 ${
                      settings.theme === 'dark' ? 'bg-gray-700' : 'bg-blue-50'
                    }`}>
                      <label className={`block text-xs font-semibold mb-1 transition-colors duration-300 ${
                        settings.theme === 'dark' ? 'text-gray-300' : 'text-blue-700'
                      }`}>📧 Email</label>
                      <p className={`text-sm font-medium transition-colors duration-300 ${
                        settings.theme === 'dark' ? 'text-white' : 'text-blue-900'
                      }`}>{userProfile.email}</p>
                    </div>
                    <div className={`rounded-xl p-4 transition-colors duration-300 ${
                      settings.theme === 'dark' ? 'bg-gray-700' : 'bg-blue-50'
                    }`}>
                      <label className={`block text-xs font-semibold mb-1 transition-colors duration-300 ${
                        settings.theme === 'dark' ? 'text-gray-300' : 'text-blue-700'
                      }`}>🏢 Department</label>
                      <p className={`text-sm font-medium transition-colors duration-300 ${
                        settings.theme === 'dark' ? 'text-white' : 'text-blue-900'
                      }`}>{userProfile.department}</p>
                    </div>
                    <div className={`rounded-xl p-4 transition-colors duration-300 ${
                      settings.theme === 'dark' ? 'bg-gray-700' : 'bg-blue-50'
                    }`}>
                      <label className={`block text-xs font-semibold mb-1 transition-colors duration-300 ${
                        settings.theme === 'dark' ? 'text-gray-300' : 'text-blue-700'
                      }`}>👔 Role</label>
                      <p className={`text-sm font-medium transition-colors duration-300 ${
                        settings.theme === 'dark' ? 'text-white' : 'text-blue-900'
                      }`}>{userProfile.role}</p>
                    </div>
                    <div className={`rounded-xl p-4 transition-colors duration-300 ${
                      settings.theme === 'dark' ? 'bg-gray-700' : 'bg-blue-50'
                    }`}>
                      <label className={`block text-xs font-semibold mb-1 transition-colors duration-300 ${
                        settings.theme === 'dark' ? 'text-gray-300' : 'text-blue-700'
                      }`}>📅 Join Date</label>
                      <p className={`text-sm font-medium transition-colors duration-300 ${
                        settings.theme === 'dark' ? 'text-white' : 'text-blue-900'
                      }`}>{new Date(userProfile.joinDate).toLocaleDateString()}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Subscription & Usage */}
            <div className={`rounded-2xl shadow-lg p-6 border transition-colors duration-300 ${
              settings.theme === 'dark' 
                ? 'bg-gray-800 border-gray-700' 
                : 'bg-white border-blue-100'
            }`}>
              <h2 className={`text-lg font-bold mb-4 flex items-center transition-colors duration-300 ${
                settings.theme === 'dark' ? 'text-white' : 'text-blue-900'
              }`}>
                <span className="mr-2">💳</span>
                Subscription & Usage
              </h2>
              
              <div className="space-y-6">
                <div className="bg-gradient-to-r from-green-50 to-green-100 p-6 rounded-xl border border-green-200">
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-sm font-semibold text-green-800">Current Plan</span>
                    <span className="bg-green-500 text-white px-3 py-1 rounded-full text-xs font-medium">
                      {paymentInfo.status}
                    </span>
                  </div>
                  <p className="text-xl font-bold text-green-700 mb-2">{paymentInfo.plan}</p>
                  <p className="text-xs text-green-600">Next billing: {new Date(paymentInfo.nextBilling).toLocaleDateString()}</p>
                </div>

                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-semibold text-blue-700">Token Usage</span>
                      <span className="text-sm text-blue-600 font-medium">{paymentInfo.usage.percentage}%</span>
                    </div>
                    <div className="w-full bg-blue-100 rounded-full h-3 mb-2">
                      <div 
                        className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-300"
                        style={{ width: `${paymentInfo.usage.percentage}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-blue-600">
                      {paymentInfo.usage.tokens.toLocaleString()} / {paymentInfo.usage.limit.toLocaleString()} tokens used
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium flex items-center justify-center">
                      <span className="mr-1">📊</span>
                      View Usage Details
                    </button>
                    <button className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium flex items-center justify-center">
                      <span className="mr-1">💳</span>
                      Manage Billing
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Status & Settings */}
          <div className="space-y-6">
            {/* Server Status */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-blue-100">
              <h2 className="text-lg font-bold text-blue-900 mb-3 flex items-center">
                <span className="mr-2">🔌</span>
                Server Status
              </h2>
              
              <div className="space-y-4">
                <div className={`flex items-center space-x-3 p-4 rounded-xl ${getStatusColor(serverStatus)}`}>
                  <span className="text-lg">{getStatusIcon(serverStatus)}</span>
                  <div>
                    <p className="text-sm font-semibold">
                      {serverStatus === 'connected' ? 'Connected' :
                       serverStatus === 'error' ? 'Disconnected' : 'Checking...'}
                    </p>
                    <p className="text-xs opacity-80">
                      {serverStatus === 'connected' ? 'All systems operational' :
                       serverStatus === 'error' ? 'Connection failed' : 'Testing connection...'}
                    </p>
                  </div>
                </div>
                
                <div className="text-xs text-blue-500 bg-blue-50 p-3 rounded-lg">
                  Status updates automatically every 30 seconds
                </div>
              </div>
            </div>

            {/* Theme Settings */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-blue-100">
              <h2 className="text-lg font-bold text-blue-900 mb-3 flex items-center">
                <span className="mr-2">🎨</span>
                Appearance
              </h2>
              
              <div className="space-y-4">
                <label className="block text-xs font-semibold text-blue-700">
                  Theme Preference
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => handleThemeChange('light')}
                    className={`p-3 rounded-lg border-2 transition-all duration-200 ${
                      settings.theme === 'light'
                        ? 'border-blue-500 bg-blue-50 text-blue-700 shadow-md'
                        : 'border-blue-200 bg-white text-blue-600 hover:border-blue-300 hover:bg-blue-50'
                    }`}
                  >
                    <div className="text-lg mb-1">☀️</div>
                    <div className="text-sm font-medium">Light</div>
                    {settings.theme === 'light' && (
                      <div className="w-2 h-2 bg-blue-500 rounded-full mx-auto mt-1"></div>
                    )}
                  </button>
                  <button
                    onClick={() => handleThemeChange('dark')}
                    className={`p-3 rounded-lg border-2 transition-all duration-200 ${
                      settings.theme === 'dark'
                        ? 'border-blue-500 bg-blue-50 text-blue-700 shadow-md'
                        : 'border-blue-200 bg-white text-blue-600 hover:border-blue-300 hover:bg-blue-50'
                    }`}
                  >
                    <div className="text-lg mb-1">🌙</div>
                    <div className="text-sm font-medium">Dark</div>
                    {settings.theme === 'dark' && (
                      <div className="w-2 h-2 bg-blue-500 rounded-full mx-auto mt-1"></div>
                    )}
                  </button>
                </div>
              </div>
            </div>


            {/* Account Actions */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-blue-100">
              <h2 className="text-lg font-bold text-blue-900 mb-3 flex items-center">
                <span className="mr-2">🔐</span>
                Account Actions
              </h2>
              
              <div className="space-y-3">
                <button className="w-full px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors text-sm font-medium flex items-center justify-center">
                  <span className="mr-2">🔑</span>
                  Change Password
                </button>
                <button className="w-full px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors text-sm font-medium flex items-center justify-center">
                  <span className="mr-2">📧</span>
                  Update Email
                </button>
                <button className="w-full px-3 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors text-sm font-medium flex items-center justify-center">
                  <span className="mr-2">🚪</span>
                  Sign Out
                </button>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}

export default Settings