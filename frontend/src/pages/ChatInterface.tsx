import React, { useState, useEffect, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import ReactMarkdown from 'react-markdown'
import { apiClient } from '../utils/api'
import { globalWebSocket, ProgressTracker as ProgressTrackerType, WebSocketStatusUpdate } from '../utils/websocket'
import { useAppContext } from '../context/AppContext'
import { ChatMessage, FileInfo, AppSettings } from '../types'
import ProgressTracker from '../components/ProgressTracker'
import StatusIndicator from '../components/StatusIndicator'

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
  const [showFolderSelection, setShowFolderSelection] = useState(false)
  const [availableFolders, setAvailableFolders] = useState<string[]>([])
  const [selectedPrompt, setSelectedPrompt] = useState<any>(null)
  const [selectedParentFolder, setSelectedParentFolder] = useState<string>('')
  const [subFolders, setSubFolders] = useState<string[]>([])
  const [showSubFolderSelection, setShowSubFolderSelection] = useState(false)
  const [activeProgress, setActiveProgress] = useState<ProgressTrackerType[]>([])
  const [statusUpdates, setStatusUpdates] = useState<WebSocketStatusUpdate[]>([])
  // Rate limiting removed from backend
  
  // SharePoint caching state
  const [sharePointCache, setSharePointCache] = useState<{[key: string]: {folders: string[], timestamp: number}}>({})
  const [isLoadingFolders, setIsLoadingFolders] = useState(false)
  const CACHE_DURATION = 5 * 60 * 1000 // 5 minutes cache
  
  // Enhanced folder and file selection for Step 2
  const [availableFiles, setAvailableFiles] = useState<any[]>([])
  const [showFileSelection, setShowFileSelection] = useState(false)
  const [selectedFolder, setSelectedFolder] = useState<string>('')
  const [isLoadingFiles, setIsLoadingFiles] = useState(false)
  const [fileCache, setFileCache] = useState<{[key: string]: {files: any[], timestamp: number}}>({})
  
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
    globalWebSocket.addProgressHandler(handleProgressUpdate)
    globalWebSocket.addStatusHandler(handleStatusUpdate)
    globalWebSocket.addErrorHandler(handleWebSocketError)
    globalWebSocket.addCloseHandler(handleWebSocketClose)

    // Cleanup handlers on unmount
    return () => {
      globalWebSocket.removeMessageHandler(handleWebSocketMessage)
      globalWebSocket.removeProgressHandler(handleProgressUpdate)
      globalWebSocket.removeStatusHandler(handleStatusUpdate)
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
                      // tokens_used removed - no longer tracked
                      // token tracking removed
        } : msg
      ))
      setIsLoading(false)
    } else if (data.type === 'error') {
      // Handle specific error types with better user guidance
      if (data.message.includes('Rate limit reached') || data.message.includes('429')) {
        toast.error('Rate limit reached. Please wait 2-3 minutes before trying again.', {
          duration: 8000,
          icon: '⏳'
        })
      } else if (data.message.includes('overloaded') || data.message.includes('529')) {
        toast.error('Server is currently overloaded. Please try again in a few minutes.', {
          duration: 10000,
          icon: '🔄'
        })
      } else if (data.message.includes('Token limit exceeded')) {
        toast.error('Token limit exceeded. Please start a new session or try a shorter query.', {
          duration: 6000,
          icon: '📊'
        })
      } else if (data.message.includes('timeout')) {
        toast.error('Request timed out. The operation may be too complex - try simplifying your query.', {
          duration: 8000,
          icon: '⏰'
        })
      } else {
        toast.error(`Error: ${data.message}`, {
          duration: 5000,
          icon: '❌'
        })
      }
      
      setMessages(prev => prev.map(msg => 
        msg.isLoading ? { ...msg, content: `Error: ${data.message}`, isLoading: false } : msg
      ))
      setIsLoading(false)
    } else if (data.type === 'connection') {
      console.log('Connection:', data.message)
    } else if (data.type === 'heartbeat') {
      console.log('Heartbeat received')
    } else if (data.type === 'timeout') {
      toast.error('Connection timeout - please reconnect', {
        duration: 5000
      })
      setIsLoading(false)
    }
  }

  const handleProgressUpdate = (progress: ProgressTrackerType) => {
    setActiveProgress(prev => {
      const existing = prev.find(p => p.id === progress.id)
      if (existing) {
        return prev.map(p => p.id === progress.id ? progress : p)
      } else {
        return [...prev, progress]
      }
    })
    
    // Rate limiting status removed from backend
  }

  const handleStatusUpdate = (status: WebSocketStatusUpdate) => {
    setStatusUpdates(prev => [...prev, status])
    console.log('Status update:', status.message)
  }

  const handleWebSocketError = (error: Event) => {
    console.error('WebSocket error:', error)
    setWsConnected(false)
    toast.error('Connection lost', {
      duration: 5000
    })
  }

  const handleWebSocketClose = () => {
    setWsConnected(false)
    toast.error('Disconnected from server', {
      duration: 5000
    })
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
      // Use WebSocket for all chat (document chat removed)
      if (globalWebSocket.getConnectionStatus()) {
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

  const fetchSharePointFolders = async (forceRefresh = false) => {
    const cacheKey = 'root'
    const now = Date.now()
    
    // Check cache first (unless force refresh)
    if (!forceRefresh && sharePointCache[cacheKey] && (now - sharePointCache[cacheKey].timestamp) < CACHE_DURATION) {
      setAvailableFolders(sharePointCache[cacheKey].folders)
      return
    }
    
    setIsLoadingFolders(true)
    try {
      const response = await apiClient.listSharePointFiles('')
      if (response.status === 'success' && response.files) {
        // Filter for folders only - backend uses 'type' field
        const folders = response.files
          .filter((item: any) => item.type === 'folder')
          .map((item: any) => item.filename)
        
        // Update cache
        setSharePointCache(prev => ({
          ...prev,
          [cacheKey]: { folders, timestamp: now }
        }))
        setAvailableFolders(folders)
      }
    } catch (error) {
      console.error('Error fetching SharePoint folders:', error)
      toast.error('Failed to fetch SharePoint folders')
    } finally {
      setIsLoadingFolders(false)
    }
  }

  const fetchSubFolders = async (parentFolder: string, forceRefresh = false) => {
    const cacheKey = parentFolder
    const now = Date.now()
    
    // Check cache first (unless force refresh)
    if (!forceRefresh && sharePointCache[cacheKey] && (now - sharePointCache[cacheKey].timestamp) < CACHE_DURATION) {
      setSubFolders(sharePointCache[cacheKey].folders)
      return sharePointCache[cacheKey].folders.length > 0
    }
    
    setIsLoadingFolders(true)
    try {
      const response = await apiClient.listSharePointFiles(parentFolder)
      if (response.status === 'success' && response.files) {
        // Filter for folders only - backend uses 'type' field
        const folders = response.files
          .filter((item: any) => item.type === 'folder')
          .map((item: any) => item.filename)
        
        // Update cache
        setSharePointCache(prev => ({
          ...prev,
          [cacheKey]: { folders, timestamp: now }
        }))
        setSubFolders(folders)
        return folders.length > 0
      }
      return false
    } catch (error) {
      console.error('Error fetching subfolders:', error)
      toast.error('Failed to fetch subfolders')
      return false
    } finally {
      setIsLoadingFolders(false)
    }
  }

  const fetchSharePointFiles = async (folderPath: string, forceRefresh = false) => {
    const cacheKey = folderPath
    const now = Date.now()
    
    // Check cache first (unless force refresh)
    if (!forceRefresh && fileCache[cacheKey] && (now - fileCache[cacheKey].timestamp) < CACHE_DURATION) {
      setAvailableFiles(fileCache[cacheKey].files)
      return
    }
    
    setIsLoadingFiles(true)
    try {
      const response = await apiClient.listSharePointFiles(folderPath)
      if (response.status === 'success' && response.files) {
        // Filter for files only (not folders) and map backend fields to frontend expectations
        const files = response.files
          .filter((item: any) => item.type !== 'folder')
          .map((item: any) => ({
            name: item.filename, // Backend returns 'filename'
            path: item.path,
            extension: item.type, // Backend returns 'type' for file extension
            size_mb: item.file_size ? (item.file_size / (1024 * 1024)).toFixed(2) : 0, // Convert bytes to MB
            modified_date: item.modified_at // Backend returns 'modified_at'
          }))
        
        // Update cache
        setFileCache(prev => ({
          ...prev,
          [cacheKey]: { files, timestamp: now }
        }))
        setAvailableFiles(files)
      }
    } catch (error) {
      console.error('Error fetching SharePoint files:', error)
      toast.error('Failed to fetch SharePoint files')
    } finally {
      setIsLoadingFiles(false)
    }
  }

  const getToolsForPrompt = (prompt: any): string[] => {
    // Check if this is Step 1 (document identification - new interactive approach)
    const isStep1 = prompt.name === 'Step1_Identifying_documents' || 
                   prompt.name === 'Step-1: Document Identification Assistant' ||
                   prompt.title === 'Step-1: Document Identification Assistant' ||
                   prompt.description.includes('identify and categorize RFP/RFI/RFQ documents')
    
    // Check if this is Step 2 (summary generation)
    const isStep2 = prompt.name === 'Step2_summarize_documents' || 
                   prompt.title === 'Step-2: Executive Summary of Procurement Document' ||
                   prompt.description.includes('Generate a clear, high-value summary')
    
    if (isStep1) {
      // For Step 1, enable only specific tools for document categorization
      const allowedTools = ['sharepoint_list_files', 'extract_pdf_text']
      const filteredTools = settings.enabledTools.filter(tool => allowedTools.includes(tool))
      console.log('Step 1 - Enabled tools:', filteredTools)
      return filteredTools
    }
    
    if (isStep2) {
      // For Step 2, disable document generation tools to prevent SharePoint uploads
      const documentGenerationTools = ['generate_pdf_document', 'generate_word_document']
      return settings.enabledTools.filter(tool => !documentGenerationTools.includes(tool))
    }
    
    // For all other prompts, use all enabled tools
    return settings.enabledTools
  }

  const handlePromptClick = async (prompt: any) => {
    console.log('Prompt clicked:', prompt)
    
    // Check if this is the Step1 or Step2 prompt template (both need folder selection)
    const isStep1 = prompt.name === 'Step1_Identifying_documents' || 
                   prompt.name === 'Step-1: Document Identification Assistant' ||
                   prompt.title === 'Step-1: Document Identification Assistant' ||
                   prompt.description.includes('identify and categorize RFP/RFI/RFQ documents')
    
    const isStep2 = prompt.name === 'Step2_summarize_documents' || 
                   prompt.title === 'Step-2: Executive Summary of Procurement Document' ||
                   prompt.description.includes('Generate a clear, high-value summary')
    
    if (isStep1 || isStep2) {
      console.log(`${isStep1 ? 'Step1' : 'Step2'} prompt detected - showing folder selection`)
      setSelectedPrompt(prompt)
      await fetchSharePointFolders()
      setShowFolderSelection(true)
      setShowPromptsPanel(false)
    } else {
      // For other prompts (including Step 3), use the description as the query
      console.log('Other prompt detected - executing directly')
      
      const promptMessage = prompt.description || prompt.name || 'Please execute this prompt template.'
      
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
            query: promptMessage,
            enabled_tools: getToolsForPrompt(prompt),
            model: settings.model
          })
          setInputMessage('')
          setShowPromptsPanel(false)
        } else {
          console.error('WebSocket not connected')
          toast.error('Connection lost. Please refresh the page.')
        }
      }, 100)
    }
  }

  const handleFolderSelection = async (folderName: string) => {
    if (selectedPrompt) {
      // Check if this is Step 1 (should work like Step 2 - with folder/file selection)
      const isStep1 = selectedPrompt.name === 'Step1_Identifying_documents' || 
                     selectedPrompt.name === 'Step-1: Document Identification Assistant' ||
                     selectedPrompt.title === 'Step-1: Document Identification Assistant' ||
                     selectedPrompt.description.includes('identify and categorize RFP/RFI/RFQ documents')
      
      // Check if this is Step 2 (needs file selection)
      const isStep2 = selectedPrompt.name === 'Step2_summarize_documents' || 
                     selectedPrompt.title === 'Step-2: Executive Summary of Procurement Document' ||
                     selectedPrompt.description.includes('Generate a clear, high-value summary')
      
      if (isStep1 || isStep2) {
        // For both Step 1 and Step 2, first check if this folder has subfolders
        const hasSubFolders = await fetchSubFolders(folderName)
        
        if (hasSubFolders) {
          // Show subfolder selection first
          setSelectedParentFolder(folderName)
          setShowSubFolderSelection(true)
          setShowFolderSelection(false)
          return
        } else {
          // No subfolders, show file selection directly
          setSelectedFolder(folderName)
          await fetchSharePointFiles(folderName)
          setShowFileSelection(true)
          setShowFolderSelection(false)
          return
        }
      }
      
      // For other prompts, check if this folder has subfolders
      const hasSubFolders = await fetchSubFolders(folderName)
      
      if (hasSubFolders) {
        // Show subfolder selection
        setSelectedParentFolder(folderName)
        setShowSubFolderSelection(true)
        setShowFolderSelection(false)
      } else {
        // No subfolders, proceed with the selected folder
        const promptMessage = `${selectedPrompt.description}\n\nPlease analyze the SharePoint folder: ${folderName}`
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
              query: promptMessage,
              enabled_tools: getToolsForPrompt(selectedPrompt),
              model: settings.model
            })
            setInputMessage('')
            setShowFolderSelection(false)
            setSelectedPrompt(null)
          }
        }, 100)
      }
    }
  }

  const handleSubFolderSelection = async (subFolderName: string) => {
    if (selectedPrompt && selectedParentFolder) {
      const fullPath = `${selectedParentFolder}/${subFolderName}`
      
      // Check if this is Step 1 (should work like Step 2 - with file selection)
      const isStep1 = selectedPrompt.name === 'Step1_Identifying_documents' || 
                     selectedPrompt.name === 'Step-1: Document Identification Assistant' ||
                     selectedPrompt.title === 'Step-1: Document Identification Assistant' ||
                     selectedPrompt.description.includes('identify and categorize RFP/RFI/RFQ documents')
      
      // Check if this is Step 2 (needs file selection)
      const isStep2 = selectedPrompt.name === 'Step2_summarize_documents' || 
                     selectedPrompt.title === 'Step-2: Executive Summary of Procurement Document' ||
                     selectedPrompt.description.includes('Generate a clear, high-value summary')
      
      if (isStep1 || isStep2) {
        // For both Step 1 and Step 2, show file selection after subfolder selection
        setSelectedFolder(fullPath)
        await fetchSharePointFiles(fullPath)
        setShowFileSelection(true)
        setShowSubFolderSelection(false)
        return
      }
      
      // For other prompts, proceed with analysis
      const promptMessage = `${selectedPrompt.description}\n\nPlease analyze the SharePoint folder: ${fullPath}`
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
            query: promptMessage,
            enabled_tools: getToolsForPrompt(selectedPrompt),
            model: settings.model
          })
          setInputMessage('')
          setShowSubFolderSelection(false)
          setSelectedPrompt(null)
          setSelectedParentFolder('')
        }
      }, 100)
    }
  }

  const handleFileSelection = (selectedFile: any) => {
    if (selectedPrompt && selectedFolder) {
      const fullPath = `${selectedFolder}/${selectedFile.name}`
      
      // Check if this is Step 1 or Step 2 to customize the message
      const isStep1 = selectedPrompt.name === 'Step1_Identifying_documents' || 
                     selectedPrompt.name === 'Step-1: Document Identification Assistant' ||
                     selectedPrompt.title === 'Step-1: Document Identification Assistant' ||
                     selectedPrompt.description.includes('identify and categorize RFP/RFI/RFQ documents')
      
      const isStep2 = selectedPrompt.name === 'Step2_summarize_documents' || 
                     selectedPrompt.title === 'Step-2: Executive Summary of Procurement Document' ||
                     selectedPrompt.description.includes('Generate a clear, high-value summary')
      
      let promptMessage = ''
      let loadingMessage = ''
      
      if (isStep1) {
        promptMessage = `${selectedPrompt.description}\n\nPlease categorize whether this document is a primary RFP document: ${selectedFile.name}\n\nFile path: ${fullPath}\n\nIMPORTANT: Use ONLY extract_pdf_text with pages="1" to read the first page. Do NOT use sharepoint_read_file.`
        loadingMessage = `Categorizing document ${selectedFile.name}...`
      } else if (isStep2) {
        promptMessage = `${selectedPrompt.description}\n\nPlease analyze the SharePoint folder: ${selectedFolder}\n\nPlease select and summarize the document: ${selectedFile.name}`
        loadingMessage = `Generating summary for ${selectedFile.name}...`
      } else {
        promptMessage = `${selectedPrompt.description}\n\nPlease analyze the SharePoint folder: ${selectedFolder}\n\nPlease select and summarize the document: ${selectedFile.name}`
        loadingMessage = `Processing ${selectedFile.name}...`
      }
      
      // Show loading toast
      toast.loading(loadingMessage, { id: 'file-summary' })
      
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
          
          const enabledTools = getToolsForPrompt(selectedPrompt)
          console.log('Step 1 - Sending message with tools:', enabledTools)
          console.log('Step 1 - Prompt message:', promptMessage)
          
          globalWebSocket.sendMessage({
            query: promptMessage,
            enabled_tools: enabledTools,
            model: settings.model
          })
          setInputMessage('')
          setShowFileSelection(false)
          setSelectedPrompt(null)
          setSelectedFolder('')
          setAvailableFiles([])
          
          // Dismiss loading toast
          toast.dismiss('file-summary')
        } else {
          console.error('WebSocket not connected')
          toast.error('Connection lost. Please refresh the page.', { id: 'file-summary' })
        }
      }, 100)
    }
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

         {/* Real-time Progress and Status Updates - Moved to bottom of sidebar */}
         {(activeProgress.length > 0 || statusUpdates.length > 0) && (
           <div className="bg-white/95 backdrop-blur-sm border-t border-blue-200/30 p-2">
             <div className="text-xs font-semibold text-blue-600 mb-2">STATUS</div>
             {/* Progress Trackers */}
             {activeProgress.map((progress) => (
               <ProgressTracker
                 key={progress.id}
                 progress={progress}
                 onComplete={() => {
                   setActiveProgress(prev => prev.filter(p => p.id !== progress.id))
                 }}
               />
             ))}
             
             {/* Status Updates */}
             {statusUpdates.map((status, index) => (
               <StatusIndicator
                 key={`${status.timestamp}-${index}`}
                 status={status}
                 onDismiss={() => {
                   setStatusUpdates(prev => prev.filter((_, i) => i !== index))
                 }}
               />
             ))}
           </div>
         )}
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
                      {/* Token usage display removed - no longer tracked */}
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
          {/* Rate limiting status indicator removed */}
          
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
                            onClick={() => handlePromptClick(prompt)}
                            className="w-full text-left p-3 bg-blue-700/10 hover:bg-blue-700/20 rounded-lg transition-colors"
                          >
                            <div className="font-medium text-blue-800 text-sm">{prompt.name}</div>
                            <div className="text-xs text-blue-600 mt-1">{prompt.description}</div>
                                                         {/* Add indicator for Step 1 */}
                             {(prompt.name === 'Step1_Identifying_documents' || 
                               prompt.name === 'Step-1: Document Identification Assistant' ||
                               prompt.title === 'Step-1: Document Identification Assistant') && (
                               <div className="text-xs text-green-600 mt-1 font-medium">
                                 🎯 Primary RFP document categorization
                               </div>
                             )}
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

                         {/* Folder Selection Modal */}
             {showFolderSelection && (
               <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                 <div className="bg-white/95 backdrop-blur-md border border-blue-100/50 rounded-xl shadow-xl p-6 max-w-md w-full mx-4">
                   <div className="flex items-center justify-between mb-4">
                     <h3 className="font-bold text-blue-800 text-lg">📁 Select SharePoint Folder</h3>
                     <div className="flex items-center space-x-2">
                       <button
                         onClick={() => fetchSharePointFolders(true)}
                         disabled={isLoadingFolders}
                         className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
                         title="Refresh folders"
                       >
                         {isLoadingFolders ? '🔄' : '🔄'} Refresh
                       </button>
                       <button
                         onClick={() => {
                           setShowFolderSelection(false)
                           setSelectedPrompt(null)
                         }}
                         className="w-6 h-6 rounded-full bg-blue-700 text-white text-xs flex items-center justify-center hover:bg-blue-800"
                       >
                         ×
                       </button>
                     </div>
                   </div>
                   
                                       <p className="text-sm text-blue-600 mb-4">
                      Choose the SharePoint folder you want to 
                      {selectedPrompt && (selectedPrompt.name === 'Step1_Identifying_documents' || 
                       selectedPrompt.name === 'Step-1: Document Identification Assistant' ||
                       selectedPrompt.title === 'Step-1: Document Identification Assistant')
                        ? ' categorize primary RFP documents from:'
                        : ' analyze for RFP/RFI/RFQ documents:'}
                    </p>
                   
                                       {/* Show special message for Step 1 */}
                    {selectedPrompt && (selectedPrompt.name === 'Step1_Identifying_documents' || 
                      selectedPrompt.name === 'Step-1: Document Identification Assistant' ||
                      selectedPrompt.title === 'Step-1: Document Identification Assistant') && (
                      <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-4">
                        <div className="flex items-start space-x-2">
                          <span className="text-green-600 text-sm">📁</span>
                                                 <div className="text-xs text-green-700">
                         <strong>Primary RFP Categorization:</strong> Step 1 will categorize whether each document is a primary RFP document or not. Select files to analyze their content.
                       </div>
                        </div>
                      </div>
                    )}
                   
                   <div className="space-y-2 max-h-60 overflow-y-auto">
                     {availableFolders.length > 0 ? (
                       availableFolders.map((folderName, index) => (
                         <button
                           key={index}
                           onClick={() => handleFolderSelection(folderName)}
                           className="w-full text-left p-3 bg-blue-700/10 hover:bg-blue-700/20 rounded-lg transition-colors border border-blue-200/50"
                         >
                           <div className="flex items-center space-x-2">
                             <span className="text-blue-600">📁</span>
                             <span className="font-medium text-blue-800">{folderName}</span>
                           </div>
                         </button>
                       ))
                     ) : (
                       <div className="text-center py-4 text-blue-600">
                         <p className="text-sm">{isLoadingFolders ? 'Loading folders...' : 'No folders found'}</p>
                         {isLoadingFolders && (
                           <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mx-auto mt-2"></div>
                         )}
                       </div>
                     )}
                   </div>
                   
                   <div className="mt-4 pt-4 border-t border-blue-200/50">
                     <button
                       onClick={() => {
                         setShowFolderSelection(false)
                         setSelectedPrompt(null)
                       }}
                       className="w-full px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
                     >
                       Cancel
                     </button>
                   </div>
                 </div>
               </div>
             )}

             {/* Subfolder Selection Modal */}
             {showSubFolderSelection && (
               <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                 <div className="bg-white/95 backdrop-blur-md border border-blue-100/50 rounded-xl shadow-xl p-6 max-w-md w-full mx-4">
                   <div className="flex items-center justify-between mb-4">
                                           <h3 className="font-bold text-blue-800 text-lg">
                        {selectedPrompt && (selectedPrompt.name === 'Step1_Identifying_documents' || 
                         selectedPrompt.name === 'Step-1: Document Identification Assistant' ||
                         selectedPrompt.title === 'Step-1: Document Identification Assistant')
                          ? '📁 Select Project Folder' 
                          : selectedPrompt && (selectedPrompt.name === 'Step2_summarize_documents' || 
                           selectedPrompt.title === 'Step-2: Executive Summary of Procurement Document' ||
                           selectedPrompt.description.includes('Generate a clear, high-value summary'))
                          ? '📁 Select Project Folder' 
                          : '📁 Select Project Folder'}
                      </h3>
                     <div className="flex items-center space-x-2">
                       <button
                         onClick={() => fetchSubFolders(selectedParentFolder, true)}
                         disabled={isLoadingFolders}
                         className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
                         title="Refresh subfolders"
                       >
                         {isLoadingFolders ? '🔄' : '🔄'} Refresh
                       </button>
                       <button
                         onClick={() => {
                           setShowSubFolderSelection(false)
                           setSelectedPrompt(null)
                           setSelectedParentFolder('')
                         }}
                         className="w-6 h-6 rounded-full bg-blue-700 text-white text-xs flex items-center justify-center hover:bg-blue-800"
                       >
                         ×
                       </button>
                     </div>
                   </div>
                   
                                       <p className="text-sm text-blue-600 mb-4">
                      {selectedPrompt && (selectedPrompt.name === 'Step1_Identifying_documents' || 
                       selectedPrompt.name === 'Step-1: Document Identification Assistant' ||
                       selectedPrompt.title === 'Step-1: Document Identification Assistant')
                         ? `Choose a project folder within ${selectedParentFolder} to browse documents:`
                         : selectedPrompt && (selectedPrompt.name === 'Step2_summarize_documents' || 
                          selectedPrompt.title === 'Step-2: Executive Summary of Procurement Document' ||
                          selectedPrompt.description.includes('Generate a clear, high-value summary'))
                         ? `Choose a project folder within ${selectedParentFolder} to browse documents:`
                         : `Choose a project folder within ${selectedParentFolder}:`
                       }
                    </p>
                   
                   <div className="space-y-2 max-h-60 overflow-y-auto">
                     {subFolders.length > 0 ? (
                       subFolders.map((folderName, index) => (
                         <button
                           key={index}
                           onClick={() => handleSubFolderSelection(folderName)}
                           className="w-full text-left p-3 bg-blue-700/10 hover:bg-blue-700/20 rounded-lg transition-colors border border-blue-200/50"
                         >
                           <div className="flex items-center space-x-2">
                             <span className="text-blue-600">📁</span>
                             <span className="font-medium text-blue-800">{folderName}</span>
                           </div>
                         </button>
                       ))
                     ) : (
                       <div className="text-center py-4 text-blue-600">
                         <p className="text-sm">{isLoadingFolders ? 'Loading subfolders...' : 'No subfolders found'}</p>
                         {isLoadingFolders && (
                           <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mx-auto mt-2"></div>
                         )}
                       </div>
                     )}
                   </div>
                   
                   <div className="mt-4 pt-4 border-t border-blue-200/50 flex space-x-2">
                     <button
                       onClick={() => {
                         setShowSubFolderSelection(false)
                         setShowFolderSelection(true)
                         setSelectedParentFolder('')
                       }}
                       className="flex-1 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
                     >
                       ← Back
                     </button>
                     <button
                       onClick={() => {
                         setShowSubFolderSelection(false)
                         setSelectedPrompt(null)
                         setSelectedParentFolder('')
                       }}
                       className="flex-1 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                     >
                       Cancel
                     </button>
                   </div>
                 </div>
               </div>
             )}

                           {/* File Selection Modal for Step 1 and Step 2 */}
              {showFileSelection && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                  <div className="bg-white/95 backdrop-blur-md border border-blue-100/50 rounded-xl shadow-xl p-6 max-w-2xl w-full mx-4">
                    <div className="flex items-center justify-between mb-4">
                                             <h3 className="font-bold text-blue-800 text-lg">
                         {selectedPrompt && (selectedPrompt.name === 'Step1_Identifying_documents' || 
                          selectedPrompt.name === 'Step-1: Document Identification Assistant' ||
                          selectedPrompt.title === 'Step-1: Document Identification Assistant')
                           ? '📄 Select Document to Categorize' 
                           : '📄 Select Document to Summarize'}
                       </h3>
                     <div className="flex items-center space-x-2">
                       <button
                         onClick={() => fetchSharePointFiles(selectedFolder, true)}
                         disabled={isLoadingFiles}
                         className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
                         title="Refresh files"
                       >
                         {isLoadingFiles ? '🔄' : '🔄'} Refresh
                       </button>
                       <button
                         onClick={() => {
                           setShowFileSelection(false)
                           setSelectedPrompt(null)
                           setSelectedFolder('')
                           setAvailableFiles([])
                         }}
                         className="w-6 h-6 rounded-full bg-blue-700 text-white text-xs flex items-center justify-center hover:bg-blue-800"
                       >
                         ×
                       </button>
                     </div>
                   </div>
                   
                                                            <p className="text-sm text-blue-600 mb-4">
                       Choose a document from <span className="font-medium">{selectedFolder}</span> to 
                       {selectedPrompt && (selectedPrompt.name === 'Step1_Identifying_documents' || 
                        selectedPrompt.name === 'Step-1: Document Identification Assistant' ||
                        selectedPrompt.title === 'Step-1: Document Identification Assistant')
                         ? ' categorize as primary RFP document or not:'
                         : ' generate an executive summary:'}
                     </p>
                   
                   <div className="space-y-2 max-h-80 overflow-y-auto">
                     {availableFiles.length > 0 ? (
                       availableFiles.map((file, index) => (
                         <button
                           key={index}
                           onClick={() => handleFileSelection(file)}
                           className="w-full text-left p-4 bg-blue-700/10 hover:bg-blue-700/20 rounded-lg transition-colors border border-blue-200/50"
                         >
                           <div className="flex items-center justify-between">
                             <div className="flex items-center space-x-3">
                               <span className="text-blue-600 text-lg">
                                 {file.extension === 'pdf' ? '📄' : 
                                  file.extension === 'docx' ? '📝' : 
                                  file.extension === 'xlsx' ? '📊' : '📄'}
                               </span>
                               <div>
                                 <div className="font-medium text-blue-800">{file.name}</div>
                                 <div className="text-xs text-blue-600 mt-1">
                                   {file.size_mb ? `${file.size_mb} MB` : 'Unknown size'} • {file.extension?.toUpperCase() || 'Unknown type'}
                                   {file.modified_date && ` • Modified: ${new Date(file.modified_date).toLocaleDateString()}`}
                                 </div>
                               </div>
                             </div>
                                                           <div className="text-blue-600 text-sm">
                                {selectedPrompt && (selectedPrompt.name === 'Step1_Identifying_documents' || 
                                 selectedPrompt.name === 'Step-1: Document Identification Assistant' ||
                                 selectedPrompt.title === 'Step-1: Document Identification Assistant')
                                  ? 'Click to categorize →'
                                  : 'Click to summarize →'}
                              </div>
                           </div>
                         </button>
                       ))
                     ) : (
                       <div className="text-center py-8 text-blue-600">
                         {isLoadingFiles ? (
                           <div>
                             <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-3"></div>
                             <p className="text-sm">Loading files from {selectedFolder}...</p>
                           </div>
                         ) : (
                           <div>
                             <p className="text-sm">No files found in this folder</p>
                             <p className="text-xs mt-2 text-blue-500">Try selecting a different folder or check folder permissions</p>
                           </div>
                         )}
                       </div>
                     )}
                   </div>
                   
                   <div className="mt-4 pt-4 border-t border-blue-200/50 flex space-x-2">
                     <button
                       onClick={() => {
                         setShowFileSelection(false)
                         setShowFolderSelection(true)
                         setSelectedFolder('')
                         setAvailableFiles([])
                       }}
                       className="flex-1 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
                     >
                       ← Back to Folders
                     </button>
                     <button
                       onClick={() => {
                         setShowFileSelection(false)
                         setSelectedPrompt(null)
                         setSelectedFolder('')
                         setAvailableFiles([])
                       }}
                       className="flex-1 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                     >
                       Cancel
                     </button>
                   </div>
                 </div>
               </div>
             )}

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

                         {/* Stop Button - Show when loading */}
             {isLoading && (
               <button
                 onClick={() => {
                   // Stop the current operation
                   setIsLoading(false)
                   // Clear any loading messages
                   setMessages(prev => prev.map(msg => 
                     msg.isLoading ? { ...msg, content: 'Operation cancelled by user', isLoading: false } : msg
                   ))
                   // Clear progress and status updates
                   setActiveProgress([])
                   setStatusUpdates([])
                   // Rate limiting status removed
                   toast.success('Operation stopped')
                 }}
                 className="px-6 py-4 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-2xl hover:from-red-700 hover:to-red-800 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 font-medium"
                 title="Stop current operation"
               >
                 ⏹️
               </button>
             )}

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