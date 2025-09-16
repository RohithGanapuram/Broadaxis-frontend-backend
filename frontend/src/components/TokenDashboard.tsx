import React, { useState, useEffect } from 'react'
import { apiClient } from '../utils/api'

interface TokenDashboardProps {
  sessionId: string
  userId?: string
  onClose: () => void
}

interface TokenUsage {
  total_tokens: number
  total_requests: number
  models: Record<string, { tokens: number; requests: number }>
}

interface QueryDetail {
  request_id: string
  model: string
  input_tokens: number
  output_tokens: number
  total_tokens: number
  task_type: string
  timestamp: string
}

const TokenDashboard: React.FC<TokenDashboardProps> = ({ sessionId, userId, onClose }) => {
  const [activeTab, setActiveTab] = useState<'session' | 'user' | 'queries'>('session')
  const [sessionUsage, setSessionUsage] = useState<TokenUsage | null>(null)
  const [userUsage, setUserUsage] = useState<any>(null)
  const [queryDetails, setQueryDetails] = useState<QueryDetail[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadTokenData()
  }, [sessionId, userId])

  const loadTokenData = async () => {
    setLoading(true)
    try {
      // Load session usage
      const sessionData = await apiClient.getDetailedSessionTokenUsage(sessionId)
      if (sessionData.status === 'success') {
        setSessionUsage(sessionData.session_usage.summary)
        setQueryDetails(sessionData.session_usage.query_details || [])
      }

      // Load user usage if userId provided
      if (userId) {
        const userData = await apiClient.getUserTokenUsage(userId)
        if (userData.status === 'success') {
          setUserUsage(userData.user_usage)
        }
      }
    } catch (error) {
      console.error('Failed to load token data:', error)
    } finally {
      setLoading(false)
    }
  }

  const formatModelName = (model: string) => {
    if (model.includes('haiku')) return 'Haiku'
    if (model.includes('sonnet')) return 'Sonnet'
    if (model.includes('opus')) return 'Opus'
    return model.split('-').pop() || model
  }

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
          <div className="flex items-center justify-center space-x-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span>Loading token data...</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-6xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center space-x-2">
            <span className="text-blue-500">📊</span>
            <span>Token Usage Dashboard</span>
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-2xl"
          >
            ×
          </button>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 mb-6 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          <button
            onClick={() => setActiveTab('session')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'session'
                ? 'bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow-sm'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            📋 Session Overview
          </button>
          {userId && (
            <button
              onClick={() => setActiveTab('user')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'user'
                  ? 'bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow-sm'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              👤 User Overview
            </button>
          )}
          <button
            onClick={() => setActiveTab('queries')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'queries'
                ? 'bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow-sm'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            🔍 Query Details
          </button>
        </div>

        {/* Content */}
        {activeTab === 'session' && sessionUsage && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {sessionUsage.total_tokens.toLocaleString()}
                </div>
                <div className="text-sm text-blue-600 dark:text-blue-400">Total Tokens</div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {sessionUsage.total_requests}
                </div>
                <div className="text-sm text-green-600 dark:text-green-400">Total Requests</div>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-200 dark:border-purple-800">
                <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {Object.keys(sessionUsage.models).length}
                </div>
                <div className="text-sm text-purple-600 dark:text-purple-400">Models Used</div>
              </div>
            </div>

            {/* Model Breakdown */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">Model Usage</h3>
              <div className="space-y-2">
                {Object.entries(sessionUsage.models).map(([model, stats]) => (
                  <div key={model} className="flex items-center justify-between p-3 bg-white dark:bg-gray-600 rounded border">
                    <div className="flex items-center space-x-3">
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {formatModelName(model)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-300">
                      <span>{stats.tokens.toLocaleString()} tokens</span>
                      <span>{stats.requests} requests</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'user' && userUsage && (
          <div className="space-y-6">
            {/* User Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {userUsage.total_tokens.toLocaleString()}
                </div>
                <div className="text-sm text-blue-600 dark:text-blue-400">Total Tokens</div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {userUsage.total_requests}
                </div>
                <div className="text-sm text-green-600 dark:text-green-400">Total Requests</div>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-200 dark:border-purple-800">
                <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {userUsage.total_sessions}
                </div>
                <div className="text-sm text-purple-600 dark:text-purple-400">Sessions</div>
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border border-orange-200 dark:border-orange-800">
                <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                  {Object.keys(userUsage.model_breakdown).length}
                </div>
                <div className="text-sm text-orange-600 dark:text-orange-400">Models</div>
              </div>
            </div>

            {/* Session Breakdown */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">Session Breakdown</h3>
              <div className="space-y-2">
                {Object.entries(userUsage.session_breakdown).map(([sessionId, stats]: [string, any]) => (
                  <div key={sessionId} className="flex items-center justify-between p-3 bg-white dark:bg-gray-600 rounded border">
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {sessionId.slice(0, 8)}...
                    </span>
                    <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-300">
                      <span>{stats.tokens.toLocaleString()} tokens</span>
                      <span>{stats.requests} requests</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'queries' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Recent Queries ({queryDetails.length})
              </h3>
            </div>
            
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {queryDetails.map((query) => (
                <div key={query.request_id} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg border">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {formatModelName(query.model)}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {formatDate(query.timestamp)}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div className="text-center">
                      <div className="font-semibold text-blue-600 dark:text-blue-400">
                        {query.total_tokens.toLocaleString()}
                      </div>
                      <div className="text-gray-500 dark:text-gray-400">Total</div>
                    </div>
                    <div className="text-center">
                      <div className="font-semibold text-green-600 dark:text-green-400">
                        {query.input_tokens.toLocaleString()}
                      </div>
                      <div className="text-gray-500 dark:text-gray-400">Input</div>
                    </div>
                    <div className="text-center">
                      <div className="font-semibold text-purple-600 dark:text-purple-400">
                        {query.output_tokens.toLocaleString()}
                      </div>
                      <div className="text-gray-500 dark:text-gray-400">Output</div>
                    </div>
                    <div className="text-center">
                      <div className="font-semibold text-orange-600 dark:text-orange-400">
                        {query.task_type}
                      </div>
                      <div className="text-gray-500 dark:text-gray-400">Type</div>
                    </div>
                  </div>
                  
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 text-center">
                    ID: {query.request_id}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default TokenDashboard

