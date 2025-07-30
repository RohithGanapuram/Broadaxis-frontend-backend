import React, { useEffect, useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import toast from 'react-hot-toast'
import Layout from './components/Layout'
import ChatInterface from './pages/ChatInterface'
import FileManager from './pages/FileManager'
import Settings from './pages/Settings'
import { apiClient } from './utils/api'
import { globalWebSocket } from './utils/websocket'
import { AppProvider, useAppContext } from './context/AppContext'

const AppContent: React.FC = () => {
  const { setTools, setPrompts, setIsConnected } = useAppContext()
  const [isInitialized, setIsInitialized] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'error'>('connecting')

  useEffect(() => {
    const initializeApp = async () => {
      try {
        toast.loading('Initializing BroadAxis-AI...', { id: 'app-init' })
        const healthCheck = await apiClient.healthCheck()
        
        if (healthCheck.status !== 'healthy') {
          throw new Error('Server not healthy')
        }
        
        toast.loading('Connecting to server...', { id: 'app-init' })
        await globalWebSocket.connect()
        setIsConnected(true)
        
        toast.loading('Loading tools and prompts...', { id: 'app-init' })
        const [toolsResponse, promptsResponse] = await Promise.all([
          apiClient.getAvailableTools(),
          apiClient.getAvailablePrompts()
        ])
        
        setTools(toolsResponse.tools)
        setPrompts(promptsResponse.prompts)
        setConnectionStatus('connected')
        setIsInitialized(true)
        toast.success(`Ready! ${toolsResponse.tools.length} tools, ${promptsResponse.prompts.length} prompts loaded`, { id: 'app-init' })
        
      } catch (error) {
        console.error('Failed to initialize app:', error)
        setConnectionStatus('error')
        toast.error('Failed to connect to server', { id: 'app-init' })
      }
    }

    initializeApp()
  }, [setTools, setPrompts, setIsConnected])

  if (!isInitialized) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">BroadAxis-AI</h2>
          <p className="text-gray-600">
            {connectionStatus === 'connecting' && 'Connecting to server...'}
            {connectionStatus === 'error' && 'Connection failed. Please refresh.'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<ChatInterface />} />
        <Route path="/files" element={<FileManager />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  )
}

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  )
}

export default App
