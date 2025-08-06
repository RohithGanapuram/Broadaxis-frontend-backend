import React, { useEffect, useState } from 'react'
import toast from 'react-hot-toast'
import Login from './components/Auth/Login'
import Register from './components/Auth/Register'
import DashboardLayout from './components/DashboardLayout'
import { apiClient } from './utils/api'
import { globalWebSocket } from './utils/websocket'
import { AppProvider, useAppContext } from './context/AppContext'
import { AuthProvider, useAuth } from './context/AuthContext'
 
const AppContent: React.FC = () => {
  const { setTools, setPrompts, setIsConnected } = useAppContext()
  const { isLoggedIn, login } = useAuth()
  const [isInitialized, setIsInitialized] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'error'>('connecting')
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login')
 
  useEffect(() => {
    const initializeApp = async () => {
      if (!isLoggedIn) {
        setIsInitialized(true)
        return
      }
 
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
        setIsInitialized(true) // Still allow access to dashboard even if chat fails
      }
    }
 
    initializeApp()
  }, [setTools, setPrompts, setIsConnected, isLoggedIn])
 
  // Show auth pages if not logged in
  if (!isLoggedIn) {
    return (
      <div className="app">
        {authMode === 'login' ? (
          <Login
            onLogin={login}
            switchToRegister={() => setAuthMode('register')}
          />
        ) : (
          <Register
            switchToLogin={() => setAuthMode('login')}
          />
        )}
      </div>
    );
  }
 
  // Show loading screen while initializing (only for logged in users)
  if (!isInitialized) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">BroadAxis-AI</h2>
          <p className="text-gray-600">
            {connectionStatus === 'connecting' && 'Connecting to server...'}
            {connectionStatus === 'error' && 'Connection failed. Chat may not work properly.'}
          </p>
        </div>
      </div>
    )
  }
 
  // Show main dashboard layout for logged in users
  return <DashboardLayout />
}
 
function App() {
  return (
    <AuthProvider>
      <AppProvider>
        <AppContent />
      </AppProvider>
    </AuthProvider>
  )
}
 
export default App