import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react'
import { Tool, Prompt, ChatMessage, ChatSession } from '../types'

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
  createNewSession: () => string
  switchToSession: (sessionId: string) => void
  deleteSession: (sessionId: string) => void
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

  // Load sessions from localStorage on mount
  useEffect(() => {
    const savedSessions = localStorage.getItem('broadaxis-chat-sessions')
    if (savedSessions) {
      const sessions = JSON.parse(savedSessions).map((s: any) => ({
        ...s,
        createdAt: new Date(s.createdAt),
        updatedAt: new Date(s.updatedAt),
        messages: s.messages.map((m: any) => ({ ...m, timestamp: new Date(m.timestamp) }))
      }))
      setChatSessions(sessions)
      if (sessions.length > 0) {
        const lastSession = sessions[sessions.length - 1]
        setCurrentSessionId(lastSession.id)
        setMessages(lastSession.messages)
      }
    }
  }, [])

  // Save sessions to localStorage whenever they change
  useEffect(() => {
    if (chatSessions.length > 0) {
      localStorage.setItem('broadaxis-chat-sessions', JSON.stringify(chatSessions))
    }
  }, [chatSessions])

  // Update current session messages when messages change
  useEffect(() => {
    if (currentSessionId && messages.length > 0) {
      setChatSessions(prev => prev.map(session => 
        session.id === currentSessionId 
          ? { ...session, messages, updatedAt: new Date(), title: generateSessionTitle(messages) }
          : session
      ))
    }
  }, [messages, currentSessionId])

  const generateSessionTitle = (messages: ChatMessage[]): string => {
    const firstUserMessage = messages.find(m => m.type === 'user')
    if (firstUserMessage) {
      return firstUserMessage.content.slice(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '')
    }
    return `Chat ${new Date().toLocaleTimeString()}`
  }

  const createNewSession = (): string => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: `New Chat ${new Date().toLocaleTimeString()}`,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    }
    setChatSessions(prev => [...prev, newSession])
    setCurrentSessionId(newSession.id)
    setMessages([])
    return newSession.id
  }

  const switchToSession = (sessionId: string) => {
    const session = chatSessions.find(s => s.id === sessionId)
    if (session) {
      setCurrentSessionId(sessionId)
      setMessages(session.messages)
    }
  }

  const deleteSession = (sessionId: string) => {
    setChatSessions(prev => prev.filter(s => s.id !== sessionId))
    if (currentSessionId === sessionId) {
      const remainingSessions = chatSessions.filter(s => s.id !== sessionId)
      if (remainingSessions.length > 0) {
        switchToSession(remainingSessions[remainingSessions.length - 1].id)
      } else {
        setCurrentSessionId(null)
        setMessages([])
      }
    }
  }

  const addMessage = (message: ChatMessage) => {
    setMessages(prev => [...prev, message])
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
      deleteSession
    }}>
      {children}
    </AppContext.Provider>
  )
}