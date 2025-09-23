import React from 'react'
import { useAuth } from '../../context/AuthContext'
import StatsCards from './StatsCards'
import RecentDocuments from './RecentDocuments'

const Dashboard: React.FC = () => {
  const { currentUser } = useAuth()

  return (
    <div className="p-6 space-y-6">
      {/* Welcome Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-2xl p-8 text-white shadow-xl">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">
              Welcome back, {currentUser?.name || 'User'}! 👋
            </h1>
            <p className="text-blue-100 text-lg">
              Ready to manage your RFP/RFQ processes efficiently
            </p>
          </div>
          <div className="hidden md:block">
            <div className="w-24 h-24 bg-white/20 rounded-full flex items-center justify-center">
              <span className="text-4xl">📊</span>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <StatsCards />

      {/* Recent Documents */}
      <RecentDocuments />

      {/* Quick Actions - Commented out for now, can be added later */}
      {/* <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="p-4 bg-blue-50 hover:bg-blue-100 rounded-xl border border-blue-200 transition-colors text-left">
            <div className="text-2xl mb-2">📄</div>
            <h3 className="font-semibold text-gray-900">Upload RFP</h3>
            <p className="text-sm text-gray-600">Upload and analyze new RFP documents</p>
          </button>
          
          <button className="p-4 bg-green-50 hover:bg-green-100 rounded-xl border border-green-200 transition-colors text-left">
            <div className="text-2xl mb-2">💬</div>
            <h3 className="font-semibold text-gray-900">Start Chat</h3>
            <p className="text-sm text-gray-600">Get AI assistance for your RFP analysis</p>
          </button>
          
          <button className="p-4 bg-purple-50 hover:bg-purple-100 rounded-xl border border-purple-200 transition-colors text-left">
            <div className="text-2xl mb-2">📊</div>
            <h3 className="font-semibold text-gray-900">View Reports</h3>
            <p className="text-sm text-gray-600">Access your RFP analysis reports</p>
          </button>
        </div>
      </div> */}
    </div>
  )
}

export default Dashboard