import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react'
import { Tool, Prompt, ChatMessage, ChatSession } from '../types'

interface SharedFile {
  filename: string
  file_size: number
  modified_at: string
  type: string
  web_url?: string
  download_url?: string
  path: string
  id?: string
}

interface SharePointCache {
  files: SharedFile[]
  lastFetchTime: number
  currentPath: string
}

interface AppContextType {
  tools: Tool[]
  prompts: Prompt[]
  setTools: (tools: Tool[]) => void
  setPrompts: (prompts: Prompt[]) => void
  isConnected: boolean
  setIsConnected: (connected: boolean) => void
  messages: ChatMessage[]
  setMessages: (messages: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => void
  addMessage: (message: ChatMessage) => void
  chatSessions: ChatSession[]
  currentSessionId: string | null
  createNewSession: () => Promise<string>
  switchToSession: (sessionId: string) => void
  deleteSession: (sessionId: string) => void
  updateSessionId: (newSessionId: string) => void
  loadUserSessions: () => Promise<void>
  // SharePoint cache
  sharePointCache: SharePointCache | null
  setSharePointCache: (cache: SharePointCache | null) => void
  clearSharePointCache: () => void
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export const useAppContext = () => {
  const context = useContext(AppContext)
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider')
  }
  return context
}

interface AppProviderProps {
  children: ReactNode
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [tools, setTools] = useState<Tool[]>([])
  const [prompts, setPrompts] = useState<Prompt[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [isSwitchingSession, setIsSwitchingSession] = useState(false)
  const [sharePointCache, setSharePointCache] = useState<SharePointCache | null>(null)

  // Initialize with empty sessions - all data comes from Redis
  useEffect(() => {
    console.log('🔄 Starting fresh with Redis-only session management')
    setChatSessions([])
    setCurrentSessionId(null)
    setMessages([])
  }, [])

  // Update current session messages when messages change (but not during session switching)
  useEffect(() => {
    if (currentSessionId && messages.length > 0 && !isSwitchingSession) {
      console.log(`🔄 Updating session ${currentSessionId} with ${messages.length} messages`)
      console.log(`🔄 Current messages:`, messages.map(m => ({ id: m.id, type: m.type, content: m.content.substring(0, 50) + '...' })))
      
      setChatSessions(prev => {
        const updated = prev.map(session => 
          session.id === currentSessionId 
            ? { 
                ...session, 
                messages, 
                updatedAt: new Date(), 
                title: generateSessionTitle(messages, session.title)
              }
            : session
        )
        console.log(`🔄 Updated sessions:`, updated.map(s => ({ id: s.id, messageCount: s.messages.length, title: s.title })))
        return updated
      })
    }
  }, [messages, currentSessionId, isSwitchingSession])

  const generateSessionTitle = (messages: ChatMessage[], currentTitle?: string): string => {
    // If we already have a meaningful title (not the default "New Chat" pattern), keep it
    if (currentTitle && !currentTitle.startsWith('New Chat') && currentTitle !== 'Untitled Chat') {
      return currentTitle
    }
    
    const firstUserMessage = messages.find(m => m.type === 'user')
    if (firstUserMessage && firstUserMessage.content.trim()) {
      return firstUserMessage.content.slice(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '')
    }
    return `Chat ${new Date().toLocaleTimeString()}`
  }

  const createNewSession = async (): Promise<string> => {
    try {
      // Create a real Redis session immediately via API
      const { apiClient } = await import('../utils/api')
      const response = await apiClient.createSession()
      const sessionId = response.session_id
      
      const newSession: ChatSession = {
        id: sessionId, // Real Redis session ID from the start
        title: `New Chat ${new Date().toLocaleTimeString()}`,
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date()
      }
      
      setChatSessions(prev => [...prev, newSession])
      setCurrentSessionId(sessionId)
      setMessages([])
      
      console.log(`🆕 Created new Redis session: ${sessionId}`)
      return sessionId
    } catch (error) {
      console.error('Failed to create Redis session:', error)
      // Fallback to temp ID if Redis fails
      const tempId = `temp_${Date.now()}`
      const newSession: ChatSession = {
        id: tempId,
        title: `New Chat ${new Date().toLocaleTimeString()}`,
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date()
      }
      setChatSessions(prev => [...prev, newSession])
      setCurrentSessionId(tempId)
      setMessages([])
      return tempId
    }
  }

  // Add function to update session ID from backend
  const updateSessionId = (newSessionId: string) => {
    // Validate session ID format
    if (!newSessionId || typeof newSessionId !== 'string') {
      console.error('Invalid session ID received:', newSessionId)
      return
    }
    
    // Update the current session with the real Redis session ID
    setChatSessions(prev => prev.map(session => 
      session.id === currentSessionId 
        ? { ...session, id: newSessionId, updatedAt: new Date() }
        : session
    ))
    setCurrentSessionId(newSessionId)
    console.log(`🔄 Updated session ID from ${currentSessionId} to: ${newSessionId}`)
  }

  const loadUserSessions = async (): Promise<void> => {
    try {
      const { apiClient } = await import('../utils/api')
      const response = await apiClient.getUserSessions()
      
      if (response.status === 'success' && response.sessions) {
        const userSessions: ChatSession[] = response.sessions.map((session: any) => ({
          id: session.id,
          title: session.title,
          messages: [], // Messages will be loaded when switching to session
          createdAt: new Date(session.created_at),
          updatedAt: new Date(session.updated_at)
        }))
        
        console.log(`📂 Loaded ${userSessions.length} user sessions from backend`)
        setChatSessions(userSessions)
      }
    } catch (error) {
      console.error('Failed to load user sessions:', error)
    }
  }

  const switchToSession = async (sessionId: string) => {
    console.log(`🔄 Switching to session: ${sessionId}`)
    console.log(`🔄 Current session before switch: ${currentSessionId}`)
    console.log(`🔄 Current messages before switch: ${messages.length}`)
    
    setIsSwitchingSession(true)
    
    // Clear messages immediately to prevent mixing
    setMessages([])
    setCurrentSessionId(sessionId)
    
    // Always load from Redis backend
    try {
      const { apiClient } = await import('../utils/api')
      const backendSession = await apiClient.getSessionInfo(sessionId)
      if (backendSession && backendSession.status === 'success' && backendSession.messages) {
        // Convert backend messages to frontend format
        const backendMessages = backendSession.messages.map((m: any, index: number) => ({
          id: m.id || `backend_${Date.now()}_${index}_${Math.random().toString(36).substr(2, 9)}`,
          type: m.role === 'user' ? 'user' : 'assistant',
          content: m.content,
          timestamp: new Date(m.timestamp),
          isLoading: false
        }))
        
        console.log(`📂 Backend messages for ${sessionId}:`, backendMessages.map((m: any) => ({ id: m.id, type: m.type, content: m.content.substring(0, 50) + '...' })))
        setMessages(backendMessages)
        console.log(`📂 Loaded ${backendMessages.length} messages from Redis for session ${sessionId}`)
      } else {
        console.log(`📂 No Redis data for session ${sessionId}`)
        setMessages([])
      }
    } catch (error) {
      console.error('Could not load session from Redis:', error)
      setMessages([])
    } finally {
      setIsSwitchingSession(false)
    }
  }

  const deleteSession = async (sessionId: string) => {
    try {
      const { apiClient } = await import('../utils/api')
      await apiClient.deleteSession(sessionId)
    } catch (e) {
      console.error('Failed to delete session on backend:', e)
    }
    setChatSessions(prev => prev.filter(s => s.id !== sessionId))
    if (currentSessionId === sessionId) {
      const remaining = chatSessions.filter(s => s.id !== sessionId)
      if (remaining.length > 0) {
        switchToSession(remaining[remaining.length - 1].id)
      } else {
        setCurrentSessionId(null)
        setMessages([])
      }
    }
  }

  const addMessage = (message: ChatMessage) => {
    setMessages(prev => [...prev, message])
  }

  const clearSharePointCache = () => {
    setSharePointCache(null)
  }

  return (
    <AppContext.Provider value={{
      tools,
      prompts,
      setTools,
      setPrompts,
      isConnected,
      setIsConnected,
      messages,
      setMessages,
      addMessage,
      chatSessions,
      currentSessionId,
      createNewSession,
      switchToSession,
      deleteSession,
      updateSessionId,
      loadUserSessions,
      sharePointCache,
      setSharePointCache,
      clearSharePointCache
    }}>
      {children}
    </AppContext.Provider>
  )
}