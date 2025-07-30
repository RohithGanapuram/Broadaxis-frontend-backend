import React, { createContext, useContext, useState, ReactNode } from 'react'
import { Tool, Prompt, ChatMessage } from '../types'

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
      addMessage
    }}>
      {children}
    </AppContext.Provider>
  )
}