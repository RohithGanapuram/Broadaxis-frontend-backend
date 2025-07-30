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
  const { tools: availableTools, prompts: availablePrompts, isConnected, messages, setMessages, addMessage } = useAppContext()
  const [settings, setSettings] = useState<AppSettings>({
    model: 'claude-3-7-sonnet-20250219',
    enabledTools: [],
    autoAnalyze: false,
    theme: 'light'
  })
  const [wsConnected, setWsConnected] = useState(false)
  const [showToolsPanel, setShowToolsPanel] = useState(false)
  
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
    setSettings(prev => ({
      ...prev,
      enabledTools: availableTools
        .filter(tool => !tool.name.includes('generate_'))
        .map(tool => tool.name)
    }))

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
        msg.isLoading ? { ...msg, content: data.message, isLoading: false } : msg
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
      if (globalWebSocket.getConnectionStatus()) {
        // Use WebSocket for real-time communication
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
        toast.loading(`Uploading ${file.name}...`, { id: file.name })
        
        const fileInfo = await apiClient.uploadFile(file)
        setUploadedFiles(prev => [...prev, fileInfo])
        
        // Add upload confirmation to chat
        const uploadMessage: ChatMessage = {
          id: Date.now().toString(),
          type: 'assistant',
          content: fileInfo.message || `Document '${file.name}' uploaded successfully. You can now ask questions about it!`,
          timestamp: new Date()
        }
        addMessage(uploadMessage)
        
        toast.success(`${file.name} uploaded`, { id: file.name })
      } catch (error) {
        console.error('Upload error:', error)
        toast.error(`Failed to upload ${file.name}`, { id: file.name })
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
    multiple: true
  })

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const toggleTool = (toolName: string) => {
    setSettings(prev => ({
      ...prev,
      enabledTools: prev.enabledTools.includes(toolName)
        ? prev.enabledTools.filter(t => t !== toolName)
        : [...prev.enabledTools, toolName]
    }))
  }



  return (
    <div className="flex h-[calc(100vh-4rem)] bg-gray-50">
      {/* Left Sidebar - Chat History & Tools */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b">
          <button 
            onClick={async () => {
              setMessages([])
              setUploadedFiles([])
              setInputMessage('')
              toast.success('New chat started')
            }}
            className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            ➕ New Chat
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4">
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold text-gray-700 mb-2">💬 Recent Chats</h3>
              {messages.length > 0 ? (
                <div className="space-y-2">
                  <div className="p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded border cursor-pointer">
                    <div className="font-medium truncate">
                      {messages[0]?.content.substring(0, 30)}...
                    </div>
                    <div className="text-xs text-gray-500">
                      {messages.length} messages
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-gray-500">No previous chats</p>
              )}
            </div>
            
            {availablePrompts.length > 0 && (
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">📝 Prompt Templates</h3>
                <div className="space-y-2">
                  {availablePrompts.map((prompt, index) => (
                    <button
                      key={index}
                      onClick={() => {
                        // Send prompt invocation message
                        const promptMessage = `Please use the "${prompt.name}" prompt template to analyze the uploaded documents.`
                        setInputMessage(promptMessage)
                        // Auto-send the message to invoke the prompt
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
                              query: `PROMPT:${prompt.name}`,
                              enabled_tools: settings.enabledTools,
                              model: settings.model
                            })
                            setInputMessage('')
                          }
                        }, 100)
                      }}
                      className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded border transition-colors"
                    >
                      <div className="font-medium">{prompt.name}</div>
                      <div className="text-xs text-gray-500 mt-1">{prompt.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="p-4 border-t">
          <div className={`text-sm px-3 py-2 rounded-lg text-center ${
            wsConnected 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {wsConnected ? `✅ Connected (${availableTools.length} tools)` : '❌ Disconnected'}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h2 className="text-lg font-semibold">Chat Interface</h2>
              <select
                value={settings.model}
                onChange={(e) => setSettings(prev => ({ ...prev, model: e.target.value }))}
                className="px-3 py-1 border border-gray-300 rounded-md text-sm"
              >
                <option value="claude-3-7-sonnet-20250219">🚀 Claude 3.7 Sonnet</option>
                <option value="claude-opus-4-20250514">⭐ Claude Opus 4</option>
                <option value="claude-sonnet-4-20250514">💎 Claude 4 Sonnet</option>
              </select>
            </div>
            
            <button
              onClick={() => setShowToolsPanel(!showToolsPanel)}
              className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-md text-sm"
            >
              🔧 Tools ({settings.enabledTools.length})
            </button>
          </div>
        </div>

        {/* Uploaded Files */}
        {uploadedFiles.length > 0 && (
          <div className="bg-blue-50 border-b border-blue-200 p-4">
            <h3 className="text-sm font-medium text-blue-800 mb-2">📎 Attached Files</h3>
            <div className="space-y-2">
              {uploadedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between bg-white p-2 rounded border">
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">📄</span>
                    <div>
                      <p className="text-sm font-medium">{file.filename}</p>
                      <p className="text-xs text-gray-500">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      removeFile(index)
                      toast.success(`${file.filename} removed`)
                    }}
                    className="text-red-500 hover:text-red-700 text-sm"
                  >
                    🗑️
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-20">
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
                  className={`max-w-3xl px-4 py-2 rounded-lg ${
                    message.type === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-white border border-gray-200'
                  }`}
                >
                  {message.isLoading ? (
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                      <span>Processing...</span>
                    </div>
                  ) : (
                    <ReactMarkdown className="prose prose-sm max-w-none">
                      {message.content}
                    </ReactMarkdown>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4">
          {/* File Drop Zone */}
          <div
            {...getRootProps()}
            className={`mb-4 p-4 border-2 border-dashed rounded-lg text-center transition-colors ${
              uploadedFiles.length >= 3
                ? 'border-gray-200 bg-gray-100 cursor-not-allowed'
                : isDragActive
                ? 'border-blue-400 bg-blue-50 cursor-pointer'
                : 'border-gray-300 hover:border-gray-400 cursor-pointer'
            }`}
            style={{ pointerEvents: uploadedFiles.length >= 3 ? 'none' : 'auto' }}
          >
            <input {...getInputProps()} />
            <p className="text-sm text-gray-600">
              {uploadedFiles.length >= 3
                ? '🚫 Maximum 3 files reached. Remove files to upload more.'
                : isDragActive
                ? 'Drop files here...'
                : `📎 Drag & drop files to upload (PDF, DOCX, TXT, MD) - Max 3 files (${uploadedFiles.length}/3)`}
            </p>
            {uploadedFiles.length > 0 && (
              <p className="text-xs text-blue-600 mt-1">
                Files will be available for questions in chat
              </p>
            )}
          </div>

          {/* Message Input */}
          <div className="flex space-x-2">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
              placeholder={uploadedFiles.length > 0 ? "Ask questions about your uploaded documents..." : "Message BroadAxis-AI..."}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
            <button
              onClick={handleSendMessage}
              disabled={isLoading || (!inputMessage.trim() && uploadedFiles.length === 0)}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Send
            </button>
          </div>
        </div>
      </div>

      {/* Right Sidebar - Tools Panel */}
      {showToolsPanel && (
        <div className="w-80 bg-white border-l border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold">Available Tools</h3>
            <button
              onClick={() => setShowToolsPanel(false)}
              className="text-gray-500 hover:text-gray-700"
            >
              ✕
            </button>
          </div>
          
          <div className="space-y-2">
            {availableTools.map((tool) => (
              <label key={tool.name} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={settings.enabledTools.includes(tool.name)}
                  onChange={() => toggleTool(tool.name)}
                  className="rounded"
                />
                <div>
                  <p className="text-sm font-medium">{tool.name}</p>
                  <p className="text-xs text-gray-500">{tool.description}</p>
                </div>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default ChatInterface