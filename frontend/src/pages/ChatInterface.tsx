import React, { useState, useEffect, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import ReactMarkdown from 'react-markdown'
import { apiClient } from '../utils/api'
import { globalWebSocket } from '../utils/websocket'
import { useAppContext } from '../context/AppContext'
import { ChatMessage, FileInfo, AppSettings } from '../types'

const ChatInterface: React.FC = () => {
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<FileInfo[]>([])
  const { tools: availableTools, prompts: availablePrompts, isConnected, messages, setMessages, addMessage, chatSessions, currentSessionId, createNewSession, switchToSession, deleteSession } = useAppContext()
  const [settings, setSettings] = useState<AppSettings>({
    model: 'claude-3-7-sonnet-20250219',
    enabledTools: [],
    autoAnalyze: false,
    theme: 'light'
  })
  const [wsConnected, setWsConnected] = useState(false)
  const [showToolsPanel, setShowToolsPanel] = useState(false)
  const [showPromptsPanel, setShowPromptsPanel] = useState(false)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Initialize chat interface - connection and data already loaded at app level
  useEffect(() => {
    // Set connection status and tools from global state
    setWsConnected(isConnected)
    
    // Load saved tool preferences or enable all tools by default
    const savedTools = localStorage.getItem('broadaxis-enabled-tools')
    if (savedTools && availableTools.length > 0) {
      const parsedTools = JSON.parse(savedTools)
      // Only use saved tools if they exist in current available tools
      const validTools = parsedTools.filter((toolName: string) => 
        availableTools.some(tool => tool.name === toolName)
      )
      setSettings(prev => ({ ...prev, enabledTools: validTools }))
    } else if (availableTools.length > 0) {
      // Enable all tools by default
      const allToolNames = availableTools.map(tool => tool.name)
      setSettings(prev => ({ ...prev, enabledTools: allToolNames }))
    }

    // Add event handlers to global WebSocket
    globalWebSocket.addMessageHandler(handleWebSocketMessage)
    globalWebSocket.addErrorHandler(handleWebSocketError)
    globalWebSocket.addCloseHandler(handleWebSocketClose)

    // Cleanup handlers on unmount
    return () => {
      globalWebSocket.removeMessageHandler(handleWebSocketMessage)
      globalWebSocket.removeErrorHandler(handleWebSocketError)
      globalWebSocket.removeCloseHandler(handleWebSocketClose)
    }
  }, [isConnected, availableTools])

  const handleWebSocketMessage = (data: any) => {
    if (data.type === 'response') {
      setMessages(prev => prev.map(msg => 
        msg.isLoading ? { 
          ...msg, 
          content: data.message, 
          isLoading: false,
          tokens_used: data.tokens_used,
          tokens_remaining: data.tokens_remaining,
          usage: data.usage
        } : msg
      ))
      setIsLoading(false)
    } else if (data.type === 'status') {
      console.log('Status:', data.message)
    } else if (data.type === 'error') {
      toast.error(data.message)
      setIsLoading(false)
    }
  }

  const handleWebSocketError = (error: Event) => {
    console.error('WebSocket error:', error)
    setWsConnected(false)
    toast.error('Connection lost')
  }

  const handleWebSocketClose = () => {
    setWsConnected(false)
    toast.error('Disconnected from server')
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() && uploadedFiles.length === 0) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    }

    const assistantMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true
    }

    addMessage(userMessage)
    addMessage(assistantMessage)
    setIsLoading(true)

    try {
      // Use hybrid document chat if files are uploaded
      if (uploadedFiles.length > 0) {
        const response = await apiClient.chatWithDocument(inputMessage, currentSessionId || 'default')
        setMessages(prev => prev.map(msg => 
          msg.isLoading ? { 
            ...msg, 
            content: response.response, 
            isLoading: false,
            tokens_used: response.tokens_used,
            tokens_remaining: response.tokens_remaining,
            usage: response.usage
          } : msg
        ))
        setIsLoading(false)
      } else if (globalWebSocket.getConnectionStatus()) {
        // Use WebSocket for regular chat
        globalWebSocket.sendMessage({
          query: inputMessage,
          enabled_tools: settings.enabledTools,
          model: settings.model
        })
      } else {
        toast.error('Not connected to server. Please refresh the page.')
        setIsLoading(false)
      }
    } catch (error) {
      console.error('Error sending message:', error)
      toast.error('Failed to send message')
      setIsLoading(false)
    }

    setInputMessage('')
  }

  const handleFileUpload = async (files: File[]) => {
    // Limit to 3 files total
    const currentFileCount = uploadedFiles.length
    const newFilesCount = files.length
    const totalFiles = currentFileCount + newFilesCount
    
    if (totalFiles > 3) {
      toast.error(`Maximum 3 files allowed. You have ${currentFileCount} files, trying to add ${newFilesCount} more.`)
      return
    }
    
    for (const file of files) {
      try {
        toast.loading(`Processing ${file.name}...`, { id: file.name })
        
        const fileInfo = await apiClient.uploadFile(file, currentSessionId || 'default')
        setUploadedFiles(prev => [...prev, fileInfo])
        
        // Add upload confirmation to chat
        const uploadMessage: ChatMessage = {
          id: Date.now().toString(),
          type: 'assistant',
          content: fileInfo.message || `Document '${file.name}' processed and ready for intelligent Q&A! 🧠`,
          timestamp: new Date()
        }
        addMessage(uploadMessage)
        
        toast.success(`${file.name} ready for Q&A`, { id: file.name })
      } catch (error) {
        console.error('Upload error:', error)
        toast.error(`Failed to process ${file.name}`, { id: file.name })
      }
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleFileUpload,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md']
    },
    multiple: true,
    noClick: true // Disable dropzone click to avoid conflicts
  })

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const toggleTool = (toolName: string) => {
    setSettings(prev => {
      const newEnabledTools = prev.enabledTools.includes(toolName)
        ? prev.enabledTools.filter(t => t !== toolName)
        : [...prev.enabledTools, toolName]
      
      // Save to localStorage
      localStorage.setItem('broadaxis-enabled-tools', JSON.stringify(newEnabledTools))
      
      return {
        ...prev,
        enabledTools: newEnabledTools
      }
    })
  }



  return (
    <div className="flex h-[calc(100vh-4.5rem)] bg-blue-50">
      {/* Chat History Sidebar */}
      <div className="w-64 bg-white/70 backdrop-blur-md border-r border-blue-200/50 flex flex-col shadow-xl">
        <div className="p-3 border-b border-blue-200/50">
          <button 
            onClick={async () => {
              // Clear current session files if any
              if (uploadedFiles.length > 0) {
                try {
                  await apiClient.clearSession(currentSessionId || 'default')
                } catch (error) {
                  console.error('Failed to clear session:', error)
                }
              }
              createNewSession()
              setUploadedFiles([])
              setInputMessage('')
              toast.success('New chat started')
            }}
            className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white px-3 py-2 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 font-medium text-sm"
          >
            ✨ New Chat
          </button>
        </div>
        
        {/* Chat Sessions List */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-2">
            <h3 className="text-xs font-semibold text-blue-600 mb-2 px-2">CHAT HISTORY</h3>
            <div className="space-y-1">
              {chatSessions.slice().reverse().map((session) => (
                <div key={session.id} className="group relative">
                  <button
                    onClick={() => switchToSession(session.id)}
                    className={`w-full text-left p-2 rounded-lg text-sm transition-colors ${
                      currentSessionId === session.id
                        ? 'bg-blue-100 text-blue-800'
                        : 'text-blue-600 hover:bg-blue-50'
                    }`}
                  >
                    <div className="truncate font-medium">{session.title}</div>
                    <div className="text-xs text-blue-400 mt-1">
                      {session.updatedAt.toLocaleDateString()}
                    </div>
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      if (confirm('Delete this chat?')) {
                        deleteSession(session.id)
                        toast.success('Chat deleted')
                      }
                    }}
                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 w-5 h-5 rounded text-xs bg-red-100 text-red-600 hover:bg-red-200 transition-all"
                  >
                    ×
                  </button>
                </div>
              ))}
              {chatSessions.length === 0 && (
                <div className="text-center text-blue-400 text-xs py-4">
                  No chat history yet
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white/70 backdrop-blur-md border-b border-blue-200/50 p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h2 className="text-lg font-semibold text-blue-800">Chat Interface</h2>
              <select
                value={settings.model}
                onChange={(e) => setSettings(prev => ({ ...prev, model: e.target.value }))}
                className="px-3 py-2 border border-blue-300/50 rounded-xl text-sm bg-white/80 text-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400"
              >
                <option value="claude-3-7-sonnet-20250219">🚀 Claude 3.7 Sonnet</option>
                <option value="claude-opus-4-20250514">⭐ Claude Opus 4</option>
                <option value="claude-sonnet-4-20250514">💎 Claude 4 Sonnet</option>
              </select>
            </div>
          </div>
        </div>

        {/* Uploaded Files - Compact Design */}
        {uploadedFiles.length > 0 && (
          <div className="bg-blue-500/10 border-b border-blue-200/50 px-4 py-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-blue-700 font-medium text-sm">📎 Files:</span>
                <div className="flex items-center space-x-1">
                  {uploadedFiles.map((file, index) => (
                    <div key={index} className="group relative">
                      <div className="flex items-center space-x-1 bg-blue-600 text-white px-3 py-1 rounded-full text-xs font-medium shadow-sm hover:shadow-md transition-all">
                        <span>📄</span>
                        <span className="max-w-20 truncate">{file.filename}</span>
                        <button
                          onClick={() => {
                            removeFile(index)
                            toast.success(`${file.filename} removed`)
                          }}
                          className="ml-1 text-blue-200 hover:text-white transition-colors"
                        >
                          ×
                        </button>
                      </div>
                      {/* Tooltip */}
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-blue-700 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                        {file.filename} ({(file.size / 1024).toFixed(1)} KB)
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <span className="text-xs text-blue-600">
                {uploadedFiles.length}/3 files
              </span>
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="text-center text-blue-500 mt-20">
              <h3 className="text-lg font-medium mb-2">Welcome to BroadAxis-AI</h3>
              <p>Upload RFP/RFQ documents and ask questions about them</p>
              {uploadedFiles.length > 0 && (
                <p className="mt-2 text-sm text-blue-600">
                  📄 {uploadedFiles.length} document(s) ready for analysis
                </p>
              )}
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-3xl px-6 py-4 rounded-2xl shadow-lg ${
                    message.type === 'user'
                      ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white'
                      : 'bg-white/80 backdrop-blur-md border border-blue-200/50'
                  }`}
                >
                  {message.isLoading ? (
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                      <span>Processing...</span>
                    </div>
                  ) : (
                    <>
                      <ReactMarkdown className="prose prose-sm max-w-none">
                        {message.content}
                      </ReactMarkdown>
                      {message.type === 'assistant' && message.tokens_used && (
                        <div className="mt-3 pt-3 border-t border-blue-200/30">
                          <div className="flex items-center justify-between text-xs text-blue-600">
                            <div className="flex items-center space-x-4">
                              <span>🔢 Tokens used: {message.tokens_used.toLocaleString()}</span>
                              {message.tokens_remaining && (
                                <span>📊 Remaining: {message.tokens_remaining.toLocaleString()}</span>
                              )}
                            </div>
                            {message.usage && (
                              <div className="flex items-center space-x-2">
                                <span>Session: {message.usage.session_used.toLocaleString()}/{message.usage.session_limit.toLocaleString()}</span>
                                <span>Daily: {message.usage.daily_used.toLocaleString()}/{message.usage.daily_limit.toLocaleString()}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area with Integrated Buttons */}
        <div className="bg-white/70 backdrop-blur-md border-t border-blue-200/50 p-4 shadow-lg relative">
          <div className="flex items-end space-x-3">
            {/* File Upload Button */}
            <div className="relative">
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.doc,.txt,.md"
                onChange={(e) => {
                  const files = Array.from(e.target.files || [])
                  if (files.length > 0) {
                    handleFileUpload(files)
                  }
                  e.target.value = '' // Reset input
                }}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={uploadedFiles.length >= 3}
              />
              <button
                className={`p-3 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl ${
                  uploadedFiles.length >= 3
                    ? 'bg-blue-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 transform hover:scale-105'
                }`}
                title={`Upload files (${uploadedFiles.length}/3)`}
                onClick={(e) => {
                  if (uploadedFiles.length < 3) {
                    const input = e.currentTarget.parentElement?.querySelector('input[type="file"]') as HTMLInputElement
                    input?.click()
                  }
                }}
              >
                <span className="text-white text-lg">📎</span>
              </button>
            </div>

            {/* Prompts Button */}
            <div className="relative">
              <button
                onClick={() => setShowPromptsPanel(!showPromptsPanel)}
                className="p-3 bg-blue-600 hover:bg-blue-700 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
                title="Prompt Templates"
              >
                <span className="text-white text-lg">📝</span>
              </button>
              
              {/* Prompts Dropdown */}
              {showPromptsPanel && (
                <div className="absolute bottom-full left-0 mb-2 w-80 bg-white/90 backdrop-blur-md border border-blue-100/50 rounded-xl shadow-xl z-50">
                  <div className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-bold text-blue-800">📝 Prompt Templates ({availablePrompts.length})</h3>
                      <button
                        onClick={() => setShowPromptsPanel(false)}
                        className="w-6 h-6 rounded-full bg-blue-700 text-white text-xs flex items-center justify-center"
                      >
                        ×
                      </button>
                    </div>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {availablePrompts.length > 0 ? (
                        availablePrompts.map((prompt, index) => (
                          <button
                            key={index}
                            onClick={() => {
                              // Create a more descriptive message that shows the prompt template is being used
                              const promptMessage = prompt.description
                              setInputMessage(promptMessage)
                              setTimeout(() => {
                                if (globalWebSocket.getConnectionStatus()) {
                                  const userMessage: ChatMessage = {
                                    id: Date.now().toString(),
                                    type: 'user',
                                    content: promptMessage,
                                    timestamp: new Date()
                                  }
                                  const assistantMessage: ChatMessage = {
                                    id: (Date.now() + 1).toString(),
                                    type: 'assistant',
                                    content: '',
                                    timestamp: new Date(),
                                    isLoading: true
                                  }
                                  addMessage(userMessage)
                                  addMessage(assistantMessage)
                                  setIsLoading(true)
                                  
                                  globalWebSocket.sendMessage({
                                    query: prompt.description,
                                    enabled_tools: settings.enabledTools,
                                    model: settings.model
                                  })
                                  setInputMessage('')
                                  setShowPromptsPanel(false)
                                }
                              }, 100)
                            }}
                            className="w-full text-left p-3 bg-blue-700/10 hover:bg-blue-700/20 rounded-lg transition-colors"
                          >
                            <div className="font-medium text-blue-800 text-sm">{prompt.name}</div>
                            <div className="text-xs text-blue-600 mt-1">{prompt.description}</div>
                          </button>
                        ))
                      ) : (
                        <div className="text-center py-4 text-blue-600">
                          <p className="text-sm">No prompt templates available</p>
                          <p className="text-xs mt-1">Check server connection</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Tools Button */}
            <div className="relative">
              <button
                onClick={() => setShowToolsPanel(!showToolsPanel)}
                className="p-3 bg-blue-700 hover:bg-blue-800 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 flex items-center space-x-1"
                title="Available Tools"
              >
                <span className="text-white text-lg">🔧</span>
                <span className="bg-white text-blue-700 px-1.5 py-0.5 rounded-full text-xs font-bold">
                  {settings.enabledTools.length}
                </span>
              </button>
              
              {/* Tools Dropdown */}
              {showToolsPanel && (
                <div className="absolute bottom-full right-0 mb-2 w-80 bg-white/90 backdrop-blur-md border border-blue-100/50 rounded-xl shadow-xl z-50">
                  <div className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-bold text-blue-800">🔧 Available Tools</h3>
                      <button
                        onClick={() => setShowToolsPanel(false)}
                        className="w-6 h-6 rounded-full bg-blue-700 text-white text-xs flex items-center justify-center"
                      >
                        ×
                      </button>
                    </div>
                    <div className="flex space-x-2 mb-3">
                      <button
                        onClick={() => {
                          const allToolNames = availableTools.map(tool => tool.name)
                          setSettings(prev => ({ ...prev, enabledTools: allToolNames }))
                          localStorage.setItem('broadaxis-enabled-tools', JSON.stringify(allToolNames))
                        }}
                        className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                      >
                        Enable All
                      </button>
                      <button
                        onClick={() => {
                          setSettings(prev => ({ ...prev, enabledTools: [] }))
                          localStorage.setItem('broadaxis-enabled-tools', JSON.stringify([]))
                        }}
                        className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                      >
                        Disable All
                      </button>
                    </div>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {availableTools.map((tool) => (
                        <label key={tool.name} className="group cursor-pointer block">
                          <div className={`p-3 rounded-lg border transition-all duration-200 ${
                            settings.enabledTools.includes(tool.name)
                              ? 'border-blue-700 bg-blue-700/10'
                              : 'border-blue-200/50 hover:border-blue-400'
                          }`}>
                            <div className="flex items-start space-x-2">
                              <input
                                type="checkbox"
                                checked={settings.enabledTools.includes(tool.name)}
                                onChange={() => toggleTool(tool.name)}
                                className="mt-0.5 w-4 h-4 rounded border-blue-300 text-blue-700 focus:ring-blue-500"
                              />
                              <div className="flex-1">
                                <p className="text-sm font-medium text-blue-800">{tool.name}</p>
                                <p className="text-xs text-blue-600 mt-1">{tool.description}</p>
                              </div>
                            </div>
                          </div>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Message Input */}
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
              placeholder={uploadedFiles.length > 0 ? "Ask questions about your uploaded documents..." : "Message BroadAxis-AI..."}
              className="flex-1 px-6 py-4 border border-blue-200/50 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white/80 backdrop-blur-md shadow-lg placeholder-blue-500"
              disabled={isLoading}
            />

            {/* Send Button */}
            <button
              onClick={handleSendMessage}
              disabled={isLoading || (!inputMessage.trim() && uploadedFiles.length === 0)}
              className="px-6 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-2xl hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 font-medium"
            >
              🚀
            </button>
          </div>
          

        </div>
      </div>


    </div>
  )
}

export default ChatInterface