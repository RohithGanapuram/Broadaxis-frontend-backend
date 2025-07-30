import React, { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import { apiClient } from '../utils/api'
import { AppSettings } from '../types'

const Settings: React.FC = () => {
  const [settings, setSettings] = useState<AppSettings>({
    model: 'claude-3-7-sonnet-20250219',
    enabledTools: [],
    autoAnalyze: false,
    theme: 'light'
  })
  const [serverStatus, setServerStatus] = useState<string>('checking')
  const [userProfile] = useState({
    name: 'John Doe',
    email: 'john.doe@broadaxis.com',
    role: 'RFP Manager',
    department: 'Business Development',
    joinDate: '2023-01-15'
  })

  useEffect(() => {
    loadSettings()
  }, [])

  const loadSettings = () => {
    const savedSettings = localStorage.getItem('broadaxis-settings')
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings))
    }
    setServerStatus('connected') // Static status
  }

  const handleModelChange = (model: string) => {
    setSettings(prev => ({ ...prev, model }))
  }

  const handleThemeChange = (theme: 'light' | 'dark') => {
    setSettings(prev => ({ ...prev, theme }))
  }

  const handleAutoAnalyzeChange = (autoAnalyze: boolean) => {
    setSettings(prev => ({ ...prev, autoAnalyze }))
  }

  const saveSettings = () => {
    localStorage.setItem('broadaxis-settings', JSON.stringify(settings))
    toast.success('Settings saved successfully')
  }

  return (
    <div className="w-full p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">⚙️ Settings</h1>
        <p className="text-gray-600">Manage your account and preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* User Profile */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">👤 User Profile</h2>
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-2xl font-bold text-blue-600">{userProfile.name.split(' ').map(n => n[0]).join('')}</span>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-gray-900">{userProfile.name}</h3>
                <p className="text-gray-600">{userProfile.role}</p>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <p className="text-sm text-gray-900">{userProfile.email}</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Department</label>
                <p className="text-sm text-gray-900">{userProfile.department}</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <p className="text-sm text-gray-900">{userProfile.role}</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Join Date</label>
                <p className="text-sm text-gray-900">{new Date(userProfile.joinDate).toLocaleDateString()}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Server Status */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">🔌 Server Status</h2>
          <div className="space-y-3">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${
                serverStatus === 'connected' ? 'bg-green-500' : 
                serverStatus === 'error' ? 'bg-red-500' : 'bg-yellow-500'
              }`}></div>
              <span className="text-sm font-medium">
                {serverStatus === 'connected' ? 'Connected' :
                 serverStatus === 'error' ? 'Disconnected' : 'Checking...'}
              </span>
            </div>
            <div className="text-xs text-gray-500">
              Status updated automatically
            </div>
          </div>
        </div>

        {/* Model Configuration */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">🤖 AI Model</h2>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Model
              </label>
              <select
                value={settings.model}
                onChange={(e) => handleModelChange(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="claude-3-7-sonnet-20250219">🚀 Claude 3.7 Sonnet</option>
                <option value="claude-opus-4-20250514">⭐ Claude Opus 4</option>
                <option value="claude-sonnet-4-20250514">💎 Claude 4 Sonnet</option>
              </select>
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="autoAnalyze"
                checked={settings.autoAnalyze}
                onChange={(e) => handleAutoAnalyzeChange(e.target.checked)}
                className="rounded"
              />
              <label htmlFor="autoAnalyze" className="ml-2 text-sm text-gray-700">
                Auto-analyze uploaded files
              </label>
            </div>
          </div>
        </div>

        {/* Theme Settings */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">🎨 Appearance</h2>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Theme
              </label>
              <div className="flex space-x-2">
                <button
                  onClick={() => handleThemeChange('light')}
                  className={`px-3 py-2 text-sm rounded-md transition-colors ${
                    settings.theme === 'light'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  ☀️ Light
                </button>
                <button
                  onClick={() => handleThemeChange('dark')}
                  className={`px-3 py-2 text-sm rounded-md transition-colors ${
                    settings.theme === 'dark'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  🌙 Dark
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Account Actions */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">🔐 Account</h2>
          <div className="space-y-3">
            <button className="w-full px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors">
              🔑 Change Password
            </button>
            <button className="w-full px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors">
              📧 Update Email
            </button>
            <button className="w-full px-3 py-2 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors">
              🚪 Sign Out
            </button>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="mt-8 flex justify-end">
        <button
          onClick={saveSettings}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          💾 Save Settings
        </button>
      </div>
    </div>
  )
}

export default Settings