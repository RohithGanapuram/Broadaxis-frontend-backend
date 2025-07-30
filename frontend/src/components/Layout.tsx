import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useAppContext } from '../context/AppContext'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation()
  const { isConnected } = useAppContext()

  const isActive = (path: string) => location.pathname === path

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      {/* Modern Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-lg border-b border-blue-100/50">
        <div className="px-6 lg:px-8">
          <div className="flex justify-between items-center h-18">
            {/* Clickable Logo */}
            <div className="flex items-center space-x-4">
              <Link to="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl flex items-center justify-center shadow-lg">
                  <span className="text-white text-xl font-bold">B</span>
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-blue-700 bg-clip-text text-transparent">
                    BroadAxis-AI
                  </h1>
                  <p className="text-xs text-blue-600 -mt-1">RFP Management Platform</p>
                </div>
              </Link>
              
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
            
            {/* Modern Navigation */}
            <nav className="flex space-x-2">
              <Link
                to="/"
                className={`px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
                  isActive('/') 
                    ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg transform scale-105' 
                    : 'text-blue-600 hover:text-blue-800 hover:bg-blue-100/60 hover:shadow-md'
                }`}
              >
                <span className="text-lg">💬</span>
                <span>Chat</span>
              </Link>
              <Link
                to="/files"
                className={`px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
                  isActive('/files') 
                    ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg transform scale-105' 
                    : 'text-blue-600 hover:text-blue-800 hover:bg-blue-100/60 hover:shadow-md'
                }`}
              >
                <span className="text-lg">📁</span>
                <span>Files</span>
              </Link>
              <Link
                to="/settings"
                className={`px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
                  isActive('/settings') 
                    ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg transform scale-105' 
                    : 'text-blue-600 hover:text-blue-800 hover:bg-blue-100/60 hover:shadow-md'
                }`}
              >
                <span className="text-lg">⚙️</span>
                <span>Settings</span>
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="">
        {children}
      </main>
    </div>
  )
}

export default Layout