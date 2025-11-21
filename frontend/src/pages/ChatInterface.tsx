import React, { useState, useEffect, useRef, useMemo } from 'react'
import remarkGfm from 'remark-gfm'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import ReactMarkdown from 'react-markdown'
import { apiClient } from '../utils/api'
import { globalWebSocket, ProgressTracker as ProgressTrackerType, WebSocketStatusUpdate } from '../utils/websocket'
import { useAppContext } from '../context/AppContext'
import { ChatMessage, FileInfo, AppSettings } from '../types'
import ProgressTracker from '../components/ProgressTracker'
import StatusIndicator from '../components/StatusIndicator'
import TokenDisplay from '../components/TokenDisplay'
import TokenDashboard from '../components/TokenDashboard'
import { parseRFPDocuments, hasRFPDocumentsTable, RFPAnalysisMetadata } from '../utils/parseRFPDocuments'


// Generate unique message IDs to prevent React key conflicts
let messageIdCounter = 0
const generateMessageId = (): string => {
  messageIdCounter++
  return `${Date.now()}_${messageIdCounter}_${Math.random().toString(36).substr(2, 9)}`
}

// Format tool names for display
const formatToolName = (toolName: string): string => {
  return toolName
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase())
    .replace(/\btool\b/gi, '')
    .trim()
}

const ChatInterface: React.FC = () => {
  const [inputMessage, setInputMessage] = useState('')
  const [textareaRef, setTextareaRef] = useState<HTMLTextAreaElement | null>(null)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<FileInfo[]>([])
  const { tools: availableTools, prompts: availablePrompts, isConnected, messages, setMessages, addMessage, chatSessions, currentSessionId, createNewSession, switchToSession, deleteSession, updateSessionId } = useAppContext()
  const [settings, setSettings] = useState<AppSettings>({
    model: 'claude-sonnet-4-5-20250929',
    enabledTools: [],
    autoAnalyze: false,
    theme: 'light'
  })
  const [wsConnected, setWsConnected] = useState(false)
  const [showToolsPanel, setShowToolsPanel] = useState(false)
  const [showPromptsPanel, setShowPromptsPanel] = useState(false)
  const [showFolderSelection, setShowFolderSelection] = useState(false)
  const [availableFolders, setAvailableFolders] = useState<string[]>([])
  const [showTokenDashboard, setShowTokenDashboard] = useState(false)
  const [selectedPrompt, setSelectedPrompt] = useState<any>(null)
  const [selectedParentFolder, setSelectedParentFolder] = useState<string>('')
  const [subFolders, setSubFolders] = useState<string[]>([])
  const [showSubFolderSelection, setShowSubFolderSelection] = useState(false)
  const [activeProgress, setActiveProgress] = useState<ProgressTrackerType[]>([])
  const [statusUpdates, setStatusUpdates] = useState<WebSocketStatusUpdate[]>([])
  const [currentToolStatus, setCurrentToolStatus] = useState<string>('')
  const [toolExecutionDetails, setToolExecutionDetails] = useState<{
    tools: string[]
    completed: number
    total: number
    currentTool: string
    step: string
  } | null>(null)
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
  const [uploadedDoc, setUploadedDoc] = useState<{
  doc_id: string;
  filename: string;
  pages: number;
  text_preview: string;
  sessionId: string; 
} | null>(null);
  
  const [spRootPath, setSpRootPath] = useState<string>('');
  const toPath = (name: string) => (spRootPath ? `${spRootPath}/${name}` : name);
  const isReviewPackagePrompt = (p: any) => {
  const name = (p?.name || '').toLowerCase();
  const desc = (p?.description || '').toLowerCase();
  return name.includes('review package') || desc.includes('review package');
};
  // Local synthetic "Review Package" template if MCP doesn't provide one
  const allPromptTemplates: any[] = useMemo(() => {
    const base: any[] = Array.isArray(availablePrompts) ? [...availablePrompts] : [];

    const alreadyExists = base.some(p => isReviewPackagePrompt(p));
    if (!alreadyExists) {
      base.push({
        id: 'review_package_local',
        name: 'Review Package',
        description:
          'Analyze a completed RFP review package (folder under /Review Package in SharePoint) and rate win probability & eligibility.',
        // content is not really used here because we route to the pipeline,
        // but we keep something reasonable for consistency
        content:
          'Use the Review Package pipeline to analyze the selected SharePoint review package folder and generate a rating, eligibility status, and key findings.'
      });
    }

    return base;
  }, [availablePrompts]);


  // Document type input for Step 4
  const [showDocumentTypeInput, setShowDocumentTypeInput] = useState(false)
  const [documentTypeInput, setDocumentTypeInput] = useState('')
  const [pendingPrompt, setPendingPrompt] = useState<any>(null)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  // If the selected prompt is for Review Package, default the root to "/Review Package"
  const rootOverride: string | undefined = isReviewPackagePrompt(selectedPrompt) ? '/Review Package' : undefined;


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
  }, [isConnected, availableTools, currentSessionId])

  // --- Cache-bust from Email page: clear SharePoint caches when Emails page finishes fetching ---
const cacheBustRef = useRef<number>(0);

useEffect(() => {
  // Check once on mount
  const v0 = Number(localStorage.getItem('sp-cache-bust') || '0');
  if (v0 > cacheBustRef.current) {
    cacheBustRef.current = v0;
    setSharePointCache({}); // clear folder cache
    setFileCache({});       // clear file cache
  }

  // Listen for updates coming from Email.tsx
  const onStorage = (e: StorageEvent) => {
    if (e.key === 'sp-cache-bust') {
      const v = Number(e.newValue || '0');
      if (v > cacheBustRef.current) {
        cacheBustRef.current = v;
        setSharePointCache({});
        setFileCache({});
      }
    }
  };
  window.addEventListener('storage', onStorage);
  return () => window.removeEventListener('storage', onStorage);
}, []);


  const handleWebSocketMessage = (data: any) => {
    console.log(`🔍 Received WebSocket message:`, data)
    
    // --- NEW: handle Review Package final result
    if (data.type === 'answer' && data.task_type === 'review_package') {
      // Make the markdown report the assistant content
      const md = data?.result?.artifacts?.markdown_report || 'No report';
      const pill = data?.result?.summary
        ? `Rating ${data.result.summary.rating}/100 · ${data.result.summary.label} · Eligibility: ${data.result.summary.eligibility_status}`
        : undefined;

      setMessages(prev => {
        const updated = [...prev];
        // Update the most recent loading assistant message
        for (let i = updated.length - 1; i >= 0; i--) {
          if (updated[i].isLoading && updated[i].type === 'assistant') {
            updated[i] = {
              ...updated[i],
              content: md,
              isLoading: false,
              // store a tiny meta if you want to show a badge elsewhere
              tokenUsage: undefined
            };
            return updated;
          }
        }
        // fallback: append a new assistant message
        updated.push({
          id: generateMessageId(),
          type: 'assistant',
          content: md,
          timestamp: new Date(),
          isLoading: false
        });
        return updated;
      });

      // Clear tool status since we’re done
      setCurrentToolStatus('');
      setToolExecutionDetails(null);
      setIsLoading(false);
      return; // stop further default handling
    }


    if (data.type === 'response') {
      // Always update session ID if provided in response (for Redis testing)
      if (data.session_id) {
        console.log(`🆕 Session ID from backend: ${data.session_id}`)
        updateSessionId(data.session_id)
      }
      
      // Process response if it matches current session OR we're transitioning temp -> real ID
      const isSameSession = (!data.session_id)
        || (data.session_id === currentSessionId)
        || (!!data.session_id && !currentSessionId)
      if (isSameSession) {
        console.log(`📝 Processing response for current session: ${data.session_id}`)
        
        // Update the most recent loading message with the response
        setMessages(prev => {
          const updatedMessages = [...prev]
          console.log(`📝 Updating messages. Total messages: ${updatedMessages.length}`)
          console.log(`📝 Response data:`, { 
            message: data.message, 
            response: data.response,
            fullData: data 
          })
          
          // Find the last loading message and update it
          let foundLoadingMessage = false
          for (let i = updatedMessages.length - 1; i >= 0; i--) {
            if (updatedMessages[i].isLoading) {
              console.log(`📝 Found loading message at index ${i}, updating with response`)
              const responseContent = data.message || data.response || 'No response received'
              console.log(`📝 Setting response content:`, responseContent.substring(0, 100) + '...')
              
              updatedMessages[i] = {
                ...updatedMessages[i],
                content: responseContent,
                isLoading: false,
                tokenUsage: data.tokens_used ? {
                  total_tokens: data.tokens_used,
                  input_tokens: data.input_tokens || 0,
                  output_tokens: data.output_tokens || 0,
                  model_used: data.model_used || 'unknown',
                  request_id: data.request_id || 'unknown'
                } : undefined
              }
              foundLoadingMessage = true
              // Clear current tool status when response is received
              setCurrentToolStatus('')
              setToolExecutionDetails(null)
              break
            }
          }
          
          if (!foundLoadingMessage) {
            console.warn(`⚠️ No loading message found to update! Total messages: ${updatedMessages.length}`)
            console.warn(`⚠️ Messages:`, updatedMessages.map(m => ({ id: m.id, type: m.type, isLoading: m.isLoading })))
            // Fallback: append a new assistant message so user sees the response
            const responseContent = data.message || data.response || 'No response received'
            updatedMessages.push({
              id: generateMessageId(),
              type: 'assistant',
              content: responseContent,
              timestamp: new Date(),
              isLoading: false,
              tokenUsage: data.tokens_used ? {
                total_tokens: data.tokens_used,
                input_tokens: data.input_tokens || 0,
                output_tokens: data.output_tokens || 0,
                model_used: data.model_used || 'unknown',
                request_id: data.request_id || 'unknown'
              } : undefined
            })
          }
          return updatedMessages
        })
        setIsLoading(false)
      } else {
        console.log(`⚠️ Ignoring response for different session: ${data.session_id} (current: ${currentSessionId})`)
      }
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
      // Clear current tool status on error
      setCurrentToolStatus('')
      setToolExecutionDetails(null)
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
    
    // Extract tool execution details from progress messages
    if (progress.type === 'tool_execution' && progress.message) {
      const toolMessage = progress.message
      
      // Parse tool execution message like "Executing 2 tools: web_search_tool, sharepoint_list_files"
      const toolMatch = toolMessage.match(/Executing (\d+) tools?: (.+)/)
      if (toolMatch) {
        const totalTools = parseInt(toolMatch[1])
        const toolNamesStr = toolMatch[2]
        
        // Parse tool names, handling "and X more..." case
        let toolNames: string[] = []
        if (toolNamesStr.includes(' and ') && toolNamesStr.includes(' more...')) {
          const beforeAnd = toolNamesStr.split(' and ')[0]
          toolNames = beforeAnd.split(', ').map(name => name.trim())
        } else {
          toolNames = toolNamesStr.split(', ').map(name => name.trim())
        }
        
        setToolExecutionDetails({
          tools: toolNames,
          completed: 0,
          total: totalTools,
          currentTool: toolNames[0] || '',
          step: progress.type
        })
      }
    }
    
    // Update progress for tool execution
    if (progress.type === 'tool_execution' && progress.progress !== undefined) {
      setToolExecutionDetails(prev => {
        if (!prev) return null
        
        const completed = Math.round((progress.progress / 100) * prev.total)
        const currentIndex = Math.min(completed, prev.tools.length - 1)
        
        return {
          ...prev,
          completed,
          currentTool: prev.tools[currentIndex] || prev.tools[prev.tools.length - 1] || ''
        }
      })
    }
    
    // Rate limiting status removed from backend
  }

  const handleStatusUpdate = (status: WebSocketStatusUpdate) => {
    setStatusUpdates(prev => [...prev, status])
    setCurrentToolStatus(status.message)
    
    // Try to extract tool information from status messages
    if (status.message) {
      // Look for patterns like "Using web_search_tool" or "Calling sharepoint_list_files"
      const toolPatterns = [
        /Using (\w+)/,
        /Calling (\w+)/,
        /Executing (\w+)/,
        /Running (\w+)/,
        /(\w+_tool)/,
        /(\w+_search)/,
        /(\w+_analysis)/
      ]
      
      for (const pattern of toolPatterns) {
        const match = status.message.match(pattern)
        if (match) {
          const toolName = match[1]
          setToolExecutionDetails(prev => prev ? ({
            ...prev,
            currentTool: toolName,
            step: 'tool_execution'
          }) : {
            tools: [toolName],
            completed: 0,
            total: 1,
            currentTool: toolName,
            step: 'tool_execution'
          })
          break
        }
      }
    }
    
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


  // Auto-expand textarea function
  const autoExpandTextarea = (textarea: HTMLTextAreaElement) => {
    textarea.style.height = 'auto'
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px' // Max height of 200px
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputMessage(e.target.value)
    autoExpandTextarea(e.target)
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() && uploadedFiles.length === 0) return

    const userMessage: ChatMessage = {
      id: generateMessageId(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    }

    const assistantMessage: ChatMessage = {
      id: generateMessageId(),
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true
    }

    // Clear any previous tool status when starting new operation
    setCurrentToolStatus('')
    setToolExecutionDetails(null)
    addMessage(userMessage)
    addMessage(assistantMessage)
    setIsLoading(true)

    try {
      // Use WebSocket for all chat (document chat removed)
      if (globalWebSocket.getConnectionStatus()) {
        // Send session_id if we have one, otherwise let backend create new session
        const messageData: any = {
          query: inputMessage,
          enabled_tools: settings.enabledTools,
          model: settings.model
        }
        
        if (currentSessionId) {
          messageData.session_id = currentSessionId
          console.log(`📤 Sending message with session_id: ${currentSessionId}`)
        } else {
          console.log(`📤 Sending message without session_id - backend will create new session`)
        }

        let finalQuery = inputMessage.trim();

        if (uploadedDoc && finalQuery) {
          // keep retrieval short for chat questions
          const retrievalQuery = finalQuery.slice(0, 160);
          const k = 4;
          const { chunks } = await apiClient.searchUploadedDoc(
            uploadedDoc.doc_id,
            uploadedDoc.sessionId, 
            retrievalQuery,
            k
          );
          const contextBlock = [
            '[BEGIN SELECTED DOC EXCERPTS]',
            ...chunks.map((c, i) => {
              const snippet = c.text.length > 800 ? c.text.slice(0, 800) + '…' : c.text;
              return `--- Excerpt ${i + 1} (pages ${c.page_start}-${c.page_end}) ---\n${snippet}`;
            }),
            '[END SELECTED DOC EXCERPTS]',
          ].join('\n');
          finalQuery = `${finalQuery}\n\n${contextBlock}`;
        }
        // then send queryToSend instead of inputMessage
        

        const toolsForThisMessage = uploadedDoc ? [] : settings.enabledTools;
        globalWebSocket.sendMessage({ query: finalQuery, enabled_tools: toolsForThisMessage, model: settings.model, session_id: currentSessionId })
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
    // Reset textarea height after sending
    if (textareaRef) {
      textareaRef.style.height = 'auto'
    }
  }

  const handleFileUpload = async (files: File[]) => {
  const file = files[0];
  if (!file) return;
  try {
    toast.loading(`Uploading ${file.name}…`, { id: 'local-upload' });
    const doc = await apiClient.uploadLocal(file, currentSessionId || 'default');
    setUploadedDoc({
   doc_id: doc.doc_id,
   filename: doc.filename,
   pages: doc.pages,
   text_preview: doc.text_preview,sessionId: currentSessionId || 'default', });
    toast.success(`${doc.filename} ready for prompts`, { id: 'local-upload' });
  } catch (e: any) {
    console.error(e);
    toast.error(`Upload failed: ${e.message || e}`);
  }
};


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

  const fetchSharePointFolders = async (forceRefresh = false, rootPathOverride?: string) => {
    const root = (typeof rootPathOverride === 'string') ? rootPathOverride : spRootPath;
    const cacheKey = `root:${root || ''}`;
    const now = Date.now();
      
    // Check cache first (unless force refresh)
    if (!forceRefresh && sharePointCache[cacheKey] && (now - sharePointCache[cacheKey].timestamp) < CACHE_DURATION) {
    setAvailableFolders(sharePointCache[cacheKey].folders);
    return;
  }
    
    setIsLoadingFolders(true);
  try {
    const response = await apiClient.listSharePointFiles(root || '');
    if (response.status === 'success' && response.files) {
      const folders = response.files
        .filter((item: any) => item.type === 'folder')
        .map((item: any) => item.filename);

      setSharePointCache(prev => ({
        ...prev,
        [cacheKey]: { folders, timestamp: now }
      }));
      setAvailableFolders(folders);
    }
  } catch (error) {
    console.error('Error fetching SharePoint folders:', error);
    toast.error('Failed to fetch SharePoint folders');
  } finally {
    setIsLoadingFolders(false);
  }
};

  // Add this helper somewhere near your cache state:
  const invalidateSharePointCache = (prefix = '') => {
    setSharePointCache(prev => {
      const copy = { ...prev }
      Object.keys(copy).forEach(k => {
        if (!prefix || k.startsWith(prefix)) delete copy[k]
      })
      return copy
    })
  }

  // Update fetchSubFolders to avoid caching empty lists:
  const fetchSubFolders = async (parentFolder: string, forceRefresh = false) => {
    const cacheKey = parentFolder
    const now = Date.now()

    if (!forceRefresh && sharePointCache[cacheKey] && (now - sharePointCache[cacheKey].timestamp) < CACHE_DURATION) {
      setSubFolders(sharePointCache[cacheKey].folders)
      return sharePointCache[cacheKey].folders.length > 0
    }

    setIsLoadingFolders(true)
    try {
      const response = await apiClient.listSharePointFiles(toPath(parentFolder));  // uses root

      if (response.status === 'success' && response.files) {
        const folders = response.files
          .filter((item: any) => item.type === 'folder')
          .map((item: any) => item.filename)

        // ✅ Only cache if we actually have folders
        if (folders.length > 0) {
          setSharePointCache(prev => ({
            ...prev,
            [cacheKey]: { folders, timestamp: now }
          }))
        }
        setSubFolders(folders)
        return folders.length > 0
      }
      return false
    } catch (e) {
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
      const response = await apiClient.listSharePointFiles(toPath(folderPath));    // uses root
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

  // Document type input handlers for Step 4
  const handleDocumentTypeSubmit = async () => {
    if (!documentTypeInput.trim() || !pendingPrompt) {
      toast.error('Please enter a document type')
      return
    }

    const documentType = documentTypeInput.trim()
    const promptMessage = `${pendingPrompt.description}\n\nPlease generate a ${documentType} based on the previous RFP analysis and BroadAxis knowledge base.`

    setInputMessage(promptMessage)
    setTimeout(() => {
      if (globalWebSocket.getConnectionStatus()) {
        const userMessage: ChatMessage = {
          id: generateMessageId(),
          type: 'user',
          content: promptMessage,
          timestamp: new Date()
        }
        const assistantMessage: ChatMessage = {
          id: generateMessageId(),
          type: 'assistant',
          content: '',
          timestamp: new Date(),
          isLoading: true
        }
        addMessage(userMessage)
        addMessage(assistantMessage)
        setIsLoading(true)
        
        const enabledTools = getToolsForPrompt(pendingPrompt)
        console.log('Step 4 - Sending message with tools:', enabledTools)
        console.log('Step 4 - Document type:', documentType)
        
        globalWebSocket.sendMessage({
          query: promptMessage,
          enabled_tools: enabledTools,
          model: settings.model,
          session_id: currentSessionId
        })
        setInputMessage('')
        setShowDocumentTypeInput(false)
        setDocumentTypeInput('')
        setPendingPrompt(null)
        
        toast.success(`Generating ${documentType}...`)
      } else {
        console.error('WebSocket not connected')
        toast.error('Connection lost. Please refresh the page.')
      }
    }, 100)
  }

  const handleDocumentTypeCancel = () => {
    setShowDocumentTypeInput(false)
    setDocumentTypeInput('')
    setPendingPrompt(null)
  }

  const handleFileSelection = async (file: any) => {
    if (!selectedPrompt || !selectedFolder) return;

    try {
      setIsLoading(true);
      toast.loading('Processing document...', { id: 'file-summary' });

      // Use the full prompt template content from MCP server
      const promptMessage = selectedPrompt.content || selectedPrompt.description || selectedPrompt.name || 'Please analyze this document';
      
      // Add file-specific context
      const fullMessage = `${promptMessage}\n\nPlease analyze the document: ${file.name} from folder: ${selectedFolder}`;

      const userMessage: ChatMessage = {
        id: generateMessageId(),
        type: 'user',
        content: fullMessage,
        timestamp: new Date()
      };

      const assistantMessage: ChatMessage = {
        id: generateMessageId(),
        type: 'assistant',
        content: '',
        timestamp: new Date(),
        isLoading: true
      };

      addMessage(userMessage);
      addMessage(assistantMessage);

      const enabledTools = getToolsForPrompt(selectedPrompt);
      console.log('File selection - Sending message with tools:', enabledTools);
      console.log('File selection - Using prompt content:', selectedPrompt.content ? 'Full template' : 'Description fallback');

      if (globalWebSocket.getConnectionStatus()) {
        globalWebSocket.sendMessage({
          query: fullMessage,
          enabled_tools: enabledTools,
          model: settings.model,
          session_id: currentSessionId
        });

        setShowFileSelection(false);
        setSelectedPrompt(null);
        setSelectedFolder('');
        setAvailableFiles([]);

        toast.dismiss('file-summary');
      } else {
        console.error('WebSocket not connected');
        toast.error('Connection lost. Please refresh the page.', { id: 'file-summary' });
      }
    } catch (error: any) {
      console.error('File selection error:', error);
      setMessages(prev => prev.map(msg =>
        msg.isLoading
          ? { ...msg, content: `❌ **Error processing document:** ${error.message || error}`, isLoading: false }
          : msg
      ));
      toast.error(`Error processing document: ${error.message}`, { id: 'file-summary' });
    } finally {
      setIsLoading(false);
    }
  }

  const getToolsForPrompt = (prompt: any): string[] => {
    // Check if this is Step 2 (summary generation)
    const isStep2 = prompt.name === 'Summarize_Document' || 
                   prompt.name === 'Step2_summarize_documents' || 
                   prompt.description.includes('Generate a clear, high-value summary')
    
    // Check if this is Step 3 (Go/No-Go recommendation)
    const isStep3 = prompt.name === 'Go_No_Go_Recommendation' || 
                   prompt.name === 'Step3_go_no_go_recommendation' || 
                   prompt.description.includes('Generate an exec-style Go/No-Go matrix')
    
    // Check if this is Step 4 (Dynamic Content Generator)
    const isStep4 = prompt.name === 'Dynamic_Content_Generator' || 
                   prompt.name === 'Dynamic Content Generator' || 
                   prompt.name === 'Step4_generate_capability_statement' ||
                   prompt.description.includes('Dynamic Document Generator')
    
    if (isStep2) {
      // For Step 2, disable document generation tools to prevent SharePoint uploads
      const documentGenerationTools = ['generate_pdf_document', 'generate_word_document']
      return settings.enabledTools.filter(tool => !documentGenerationTools.includes(tool))
    }
    
    if (isStep3) {
      // For Step 3, disable document generation tools to force analysis in chat
      const documentGenerationTools = ['generate_pdf_document', 'generate_word_document']
      const filteredTools = settings.enabledTools.filter(tool => !documentGenerationTools.includes(tool))
      console.log('Step 3 - Disabled document generation tools, enabled tools:', filteredTools)
      return filteredTools
    }
    
    if (isStep4) {
      // For Step 4, enable knowledge search and document generation tools
      const allowedTools = ['Broadaxis_knowledge_search', 'generate_pdf_document', 'generate_word_document']
      const filteredTools = settings.enabledTools.filter(tool => allowedTools.includes(tool))
      console.log('Step 4 - Enabled knowledge search and document generation tools:', filteredTools)
      return filteredTools
    }
    
    // For all other prompts, use all enabled tools
    return settings.enabledTools
  }

  async function runPromptWithUploadedDoc(prompt: any) {
    if (!uploadedDoc) return;
    const retrievalQuery =
      (prompt?.description?.slice(0, 120) ?? prompt?.name ?? 'overview');

    // fetch relevant chunks to keep tokens safe
    const k = 5; // ↓ keep inputs small
    const { chunks } = await apiClient.searchUploadedDoc(
      uploadedDoc.doc_id,
      uploadedDoc.sessionId,
      retrievalQuery,
      k
    );

    let excerpts = chunks || [];
  if (!excerpts.length) {
    // optional: add a client helper in api.ts:
    // getUploadedText(docId, sessionId, startPage=1, endPage=3)
    try {
      const { text } = await apiClient.getUploadedText(
        uploadedDoc.doc_id,
        uploadedDoc.sessionId,
        1, 3
      );
      if (text?.trim()) {
        excerpts = [{ page_start: 1, page_end: 3, text }];
      }
    } catch (_) {
      // swallow; we’ll proceed with no excerpts if backend is unavailable
    }
  }

    // trim each excerpt to ~1200 chars to keep tokens low
    const contextBlock = [
      '[BEGIN SELECTED DOC EXCERPTS]',
      ...chunks.map((c, i) => {
        const snippet = c.text.length > 1200 ? c.text.slice(0, 1200) + '…' : c.text;
        return `--- Excerpt ${i + 1} (pages ${c.page_start}-${c.page_end}) ---\n${snippet}`;
      }),
      '[END SELECTED DOC EXCERPTS]',
    ].join('\n');


    const promptMessage = `${prompt.description || prompt.name}\n\n${contextBlock}`;

    // your existing websocket send flow
    const userMessage = {
      id: generateMessageId(),
      type: 'user' as const,
      content: promptMessage,
      timestamp: new Date()
    };
    const assistantMessage = {
      id: generateMessageId(),
      type: 'assistant' as const,
      content: '',
      timestamp: new Date(),
      isLoading: true
    };
    addMessage(userMessage);
    addMessage(assistantMessage);

    globalWebSocket.sendMessage({
      query: promptMessage,
      enabled_tools: getToolsForPrompt(prompt),
      model: settings.model,
      session_id: currentSessionId
      // optional: include a hint
      // , metadata: { uploaded_doc_id: uploadedDoc.doc_id }
    });

    setShowPromptsPanel(false);
  }


  const handlePromptClick = async (prompt: any) => {
    console.log('Prompt clicked:', prompt)

    // 🔹 Detect prompt types
    const isIntelligentRFP =
      prompt.name === 'Intelligent_RFP_Processing' ||
      prompt.description.includes('intelligent RFP processing')

    const isStep2 =
      prompt.name === 'Summarize_Document' ||
      prompt.name === 'Step2_summarize_documents' ||
      prompt.description.includes('Generate a clear, high-value summary')

    const isStep4 =
      prompt.name === 'Dynamic_Content_Generator' ||
      prompt.name === 'Dynamic Content Generator' ||
      prompt.name === 'Step4_generate_capability_statement' ||
      prompt.description.includes('Dynamic Document Generator') ||
      prompt.description.includes('Generate high-quality capability statements')

    // 🔹 NEW: detect Review Package prompt
    const isReviewPackage = isReviewPackagePrompt(prompt)

    // 🔹 SPECIAL CASE: Review Package prompt → use "/Review Package" root
    if (isReviewPackage) {
      console.log('Review Package prompt detected - using /Review Package root')
      
      // point SharePoint browsing to /Review Package
      setSpRootPath('/Review Package')

      // (optional but recommended) clear any old cache for other roots
      invalidateSharePointCache()

      setSelectedPrompt(prompt)
      await fetchSharePointFolders(true, '/Review Package') // force refresh at that root
      setShowFolderSelection(true)
      setShowPromptsPanel(false)
      return
    }

    // 🔹 For ALL OTHER prompts, reset root back to default
    setSpRootPath('')        // this makes fetchSharePointFolders() use the normal root
    invalidateSharePointCache()

    // ==== EXISTING LOGIC BELOW (unchanged) ====

    // If we have an uploaded file, offer to run the selected prompt on it
    if (uploadedDoc) {
      const runOnUploaded = window.confirm(
        `Run "${prompt.name}" on uploaded file "${uploadedDoc.filename}"?\n\n` +
        `OK = Uploaded file, Cancel = pick a SharePoint folder`
      )

      if (runOnUploaded) {
        await runPromptWithUploadedDoc(prompt)
        setSelectedPrompt(null)
        return
      }

      setSelectedPrompt(prompt)
      await fetchSharePointFolders()   // now this will use root = '' again
      setShowFolderSelection(true)
      setShowPromptsPanel(false)
      return
    }

    if (isIntelligentRFP) {
      console.log('Intelligent RFP processing prompt detected - showing folder selection')
      setSelectedPrompt(prompt)
      await fetchSharePointFolders()
      setShowFolderSelection(true)
      setShowPromptsPanel(false)
    } else if (isStep2) {
      console.log('Step2 prompt detected - showing folder selection')
      setSelectedPrompt(prompt)
      await fetchSharePointFolders()
      setShowFolderSelection(true)
      setShowPromptsPanel(false)
    } else if (isStep4) {
      console.log('Step 4 prompt detected - showing document type input')
      setPendingPrompt(prompt)
      setDocumentTypeInput('')
      setShowDocumentTypeInput(true)
      setShowPromptsPanel(false)
    } else {
      console.log('Other prompt detected - executing directly')

      const promptMessage =
        prompt.content ||
        prompt.description ||
        prompt.name ||
        'Please execute this prompt template.'

      setInputMessage(promptMessage)

      setTimeout(() => {
        if (globalWebSocket.getConnectionStatus()) {
          const userMessage: ChatMessage = {
            id: generateMessageId(),
            type: 'user',
            content: promptMessage,
            timestamp: new Date()
          }
          const assistantMessage: ChatMessage = {
            id: generateMessageId(),
            type: 'assistant',
            content: '',
            timestamp: new Date(),
            isLoading: true
          }

          addMessage(userMessage)
          addMessage(assistantMessage)

          globalWebSocket.sendMessage({
            query: promptMessage,
            enabled_tools: getToolsForPrompt(prompt),
            model: settings.model,
            session_id: currentSessionId
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
    if (!selectedPrompt) return;

    // Safe/normalized description
    const desc = (selectedPrompt.description || '').toLowerCase();

    // Prompt kind checks
    const isIntelligentRFP =
      selectedPrompt.name === 'Intelligent_RFP_Processing' ||
      desc.includes('intelligent rfp processing');

    const isStep2 =
      selectedPrompt.name === 'Summarize_Document' ||
      selectedPrompt.name === 'Step2_summarize_documents' ||
      desc.includes('generate a clear, high-value summary');

    if (isIntelligentRFP) {
      // Check for subfolders
      const hasSubFolders = await fetchSubFolders(folderName);

      if (hasSubFolders) {
        setSelectedParentFolder(folderName);
        setShowSubFolderSelection(true);
        setShowFolderSelection(false);
        return;
      }

      // No subfolders → run intelligent RFP processing on the folder
      try {
        setIsLoading(true);
        toast.loading('Starting intelligent RFP processing...', { id: 'intelligent-rfp' });

        const userMessage: ChatMessage = {
          id: generateMessageId(),
          type: 'user',
          content: `Process RFP folder intelligently: ${folderName}`,
          timestamp: new Date(),
        };

        const assistantMessage: ChatMessage = {
          id: generateMessageId(),
          type: 'assistant',
          content: '',
          timestamp: new Date(),
          isLoading: true,
        };

        addMessage(userMessage);
        addMessage(assistantMessage);

        const response = await apiClient.processRFPFolderIntelligent(
          folderName,
          currentSessionId || 'default'
        );

        setMessages(prev =>
          prev.map(msg =>
            msg.id === assistantMessage.id
              ? {
                  ...msg,
                  content: response.summary || response.response || 'No response received',
                  isLoading: false,
                  tokenUsage: response.token_breakdown
                    ? {
                        total_tokens: response.token_breakdown.total_tokens,
                        input_tokens: response.token_breakdown.input_tokens,
                        output_tokens: response.token_breakdown.output_tokens,
                        model_used: response.token_breakdown.model_used,
                        request_id: `rfp-${Date.now()}`,
                      }
                    : undefined,
                }
              : msg
          )
        );

        toast.success('Intelligent RFP processing completed!', { id: 'intelligent-rfp' });
      } catch (error: any) {
        console.error('Intelligent RFP processing error:', error);
        setMessages(prev =>
          prev.map(msg =>
            msg.isLoading
              ? { ...msg, content: `❌ **Error processing RFP folder:** ${error.message || error}`, isLoading: false }
              : msg
          )
        );
        toast.error(`Intelligent RFP processing error: ${error.message}`, { id: 'intelligent-rfp' });
      } finally {
        setIsLoading(false);
        setShowFolderSelection(false);
        setSelectedPrompt(null);
      }
      return;
    } else if (isStep2) {
      // Step 2 (Summarize): use the Subfolder modal and show BOTH subfolders + files for this folder
      await fetchSubFolders(folderName); // or await fetchSubFolders(folderName, true) for hard refresh
      setSelectedParentFolder(folderName);
      setSelectedFolder(folderName);
      setShowSubFolderSelection(true);
      setShowFolderSelection(false);
      await fetchSharePointFiles(folderName, true); // load files into same modal
      return; // ⬅ important: don't fall through to "other prompts"
    }

    // --- Other prompts (unchanged behavior) ---
    const hasSubFolders = await fetchSubFolders(folderName);

    if (hasSubFolders) {
      setSelectedParentFolder(folderName);
      setShowSubFolderSelection(true);
      setShowFolderSelection(false);
    } else {
      const promptMessage = `${selectedPrompt.description}\n\nPlease analyze the SharePoint folder: ${folderName}`;
      setInputMessage(promptMessage);

      setTimeout(() => {
        if (globalWebSocket.getConnectionStatus()) {
          const userMessage: ChatMessage = {
            id: generateMessageId(),
            type: 'user',
            content: promptMessage,
            timestamp: new Date(),
          };
          const assistantMessage: ChatMessage = {
            id: generateMessageId(),
            type: 'assistant',
            content: '',
            timestamp: new Date(),
            isLoading: true,
          };
          addMessage(userMessage);
          addMessage(assistantMessage);
          setIsLoading(true);

          globalWebSocket.sendMessage({
            query: promptMessage,
            enabled_tools: getToolsForPrompt(selectedPrompt),
            model: settings.model,
          });

          setInputMessage('');
          setShowFolderSelection(false);
          setSelectedPrompt(null);
        }
      }, 100);
    }
  };
  
  const handleSubFolderSelection = async (subFolderName: string) => {
    if (!selectedPrompt || !selectedParentFolder) return;

    const fullPath = `${selectedParentFolder}/${subFolderName}`;

    // --- Prompt kind detection (keep your existing names/phrases) ---
    const name = (selectedPrompt.name || '').toLowerCase();
    const desc = (selectedPrompt.description || '').toLowerCase();

    const isIntelligentRFP =
      name.includes('intelligent_rfp_processing') ||
      desc.includes('intelligent rfp processing');

    const isStep1 =
      name.includes('step1') ||
      name.includes('step-1') ||
      desc.includes('identify and categorize rfp');

    const isStep2 =
      name.includes('step2') ||
      name.includes('step-2') ||
      name.includes('summarize') ||
      name.includes('summary') ||
      desc.includes('generate a clear, high-value summary') ||
      desc.includes('summarize') ||
      desc.includes('summary');

    const isStep3 =
      name.includes('step3') ||
      name.includes('step-3') ||
      desc.includes('go/no-go');

    const isStep4 =
      name.includes('step4') ||
      name.includes('step-4') ||
      desc.includes('proposal') ||
      desc.includes('capability statement');

    const isStep5 =
      name.includes('step5') ||
      name.includes('step-5') ||
      desc.includes('fill missing information');

    // --- NEW: drill-down check for ALL prompts ---
    const hasMore = await fetchSubFolders(fullPath, true);
    if (hasMore) {
      setSelectedParentFolder(fullPath);
      setShowSubFolderSelection(true);

      // ✅ Step 2: also load files for this folder so the modal shows both lists
      if (isStep2) {
        setSelectedFolder(fullPath);
        await fetchSharePointFiles(fullPath, true);
      }
      return;
    }


    // --- Leaf folder reached: act per prompt type ---

    // Intelligent RFP → run folder processing API
    if (isIntelligentRFP) {
      try {
        setIsLoading(true);
        toast.loading('Starting intelligent RFP processing...', { id: 'intelligent-rfp' });

        const userMessage: ChatMessage = {
          id: generateMessageId(),
          type: 'user',
          content: `Process RFP folder intelligently: ${fullPath}`,
          timestamp: new Date()
        };

        const assistantMessage: ChatMessage = {
          id: generateMessageId(),
          type: 'assistant',
          content: '',
          timestamp: new Date(),
          isLoading: true
        };

        addMessage(userMessage);
        addMessage(assistantMessage);

        const response = await apiClient.processRFPFolderIntelligent(fullPath, currentSessionId || 'default');

        setMessages(prev => prev.map(msg =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content: response.summary || response.response || 'No response received',
                isLoading: false,
                tokenUsage: response.token_breakdown ? {
                  total_tokens: response.token_breakdown.total_tokens,
                  input_tokens: response.token_breakdown.input_tokens,
                  output_tokens: response.token_breakdown.output_tokens,
                  model_used: response.token_breakdown.model_used,
                  request_id: `rfp-${Date.now()}`
                } : undefined
              }
            : msg
        ));

        toast.success('Intelligent RFP processing completed!', { id: 'intelligent-rfp' });
      } catch (error: any) {
        console.error('Intelligent RFP processing error:', error);
        setMessages(prev => prev.map(msg =>
          msg.isLoading
            ? { ...msg, content: `❌ **Error processing RFP folder:** ${error.message || error}`, isLoading: false }
            : msg
        ));
        toast.error(`Intelligent RFP processing error: ${error.message}`, { id: 'intelligent-rfp' });
      } finally {
        setIsLoading(false);
        setShowSubFolderSelection(false);
        setSelectedPrompt(null);
        setSelectedParentFolder('');
      }
      return;
    }

    // Step 1 & Step 2 → open file selection modal for this leaf folder
   // Step 1 → open file selection modal (unchanged)
    if (isStep1) {
      setSelectedFolder(fullPath);
      await fetchSharePointFiles(fullPath);
      setShowFileSelection(true);
      setShowSubFolderSelection(false);
      return;
    }

    // Step 2 → stay in this modal, but refresh listing and files here
    if (isStep2) {
      setSelectedParentFolder(fullPath);       // drill deeper but keep modal open
      setSelectedFolder(fullPath);
      await fetchSubFolders(fullPath, true);   // refresh subfolders
      await fetchSharePointFiles(fullPath, true);  // refresh files
      setShowSubFolderSelection(true);
      return;
    }


    // Step 3 / Step 4 / Step 5 / Others → send a message targeting this folder
    // (If you later want Step 4 to ask for "document type", hook your existing showDocumentTypeInput here.)
    const promptMessage = `${selectedPrompt.description}\n\nPlease analyze the SharePoint folder: ${fullPath}`;
    setInputMessage(promptMessage);

    setTimeout(() => {
      if (globalWebSocket.getConnectionStatus()) {
        const userMessage: ChatMessage = {
          id: generateMessageId(),
          type: 'user',
          content: promptMessage,
          timestamp: new Date()
        };
        const assistantMessage: ChatMessage = {
          id: generateMessageId(),
          type: 'assistant',
          content: '',
          timestamp: new Date(),
          isLoading: true
        };
        addMessage(userMessage);
        addMessage(assistantMessage);
        setIsLoading(true);

        globalWebSocket.sendMessage({
          query: promptMessage,
          enabled_tools: getToolsForPrompt(selectedPrompt),
          model: settings.model,
          session_id: currentSessionId
        });

        setInputMessage('');
        setShowSubFolderSelection(false);
        setSelectedPrompt(null);
        setSelectedParentFolder('');
      }
    }, 100);
  };

  // Ensure a parent RFP task exists, return its id
  async function ensureParentRfpTask(rfpPath: string, folderName: string) {
    // 1) Try to find an existing parent RFP/RFI/RFQ task by document path or title
    const allTasks = await apiClient.getTasks();
    const existing = allTasks.find((t: any) => {
      if (t.category !== 'Project') return false;
      if (!['RFP', 'RFI', 'RFQ'].includes(t.type)) return false;

      // Match by exact document path (case-insensitive) OR by title containing folder name
      const docMatch =
        (t.document && t.document.toLowerCase() === rfpPath.toLowerCase());
      const titleMatch =
        (t.title && (t.title === folderName || t.title.includes(folderName)));

      return docMatch || titleMatch;
    });

    if (existing) return existing.id;

    // 2) If none, create a new parent RFP task
    const created = await apiClient.createTask({
      category: 'Project',
      type: 'RFP',
      title: folderName,
      document: rfpPath,
      status: 'Assigned',
      priority: 'Medium',
      // assigned_to and due_date can be filled later on the Dashboard;
      // children can inherit when you import (once parent has them)
    });
    return created.id;
  }




  return (
    <div className="flex h-[calc(100vh-4.5rem)] bg-blue-50">
      {/* Chat History Sidebar */}
      <div className={`${isSidebarCollapsed ? 'w-16' : 'w-64'} transition-all duration-300 ease-in-out bg-white/70 backdrop-blur-md border-r border-blue-200/50 flex flex-col shadow-xl`}>
        <div className="p-3 border-b border-blue-200/50 space-y-2">
          {/* Sidebar Toggle Button */}
          <div className="flex items-center justify-between mb-2">
            {!isSidebarCollapsed && (
              <div className="flex items-center space-x-2">
                <div className="w-6 h-6 bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg flex items-center justify-center shadow-lg">
                  <span className="text-white text-xs font-bold">💬</span>
                </div>
                <h3 className="text-sm font-bold bg-gradient-to-r from-blue-600 to-blue-700 bg-clip-text text-transparent">
                  Chat Sessions
                </h3>
              </div>
            )}
            <button
              onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
              className="p-1.5 rounded-lg bg-white/80 hover:bg-white shadow-md transition-all duration-200 hover:scale-105"
            >
              <span className="text-gray-600 text-sm">
                {isSidebarCollapsed ? '→' : '←'}
              </span>
            </button>
          </div>
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
              await createNewSession()
              setUploadedFiles([])
              setInputMessage('')
              toast.success('New chat started')
            }}
            className={`w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white px-3 py-2 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 font-medium text-sm flex items-center justify-center space-x-2 ${isSidebarCollapsed ? 'px-2' : 'px-3'}`}
          >
            {!isSidebarCollapsed && <span>New Chat</span>}
          </button>
          
          <button 
            onClick={() => {
              setShowTokenDashboard(true)
            }}
            className={`w-full bg-gradient-to-r from-green-600 to-green-700 text-white px-3 py-2 rounded-lg hover:from-green-700 hover:to-green-800 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 font-medium text-sm flex items-center justify-center space-x-2 ${isSidebarCollapsed ? 'px-2' : 'px-3'}`}
          >
            <span className="text-sm">📊</span>
            {!isSidebarCollapsed && <span>Token Status</span>}
          </button>
        </div>
        
                 {/* Chat Sessions List */}
         <div className="flex-1 overflow-y-auto">
           <div className="p-2">
             {!isSidebarCollapsed && (
               <h3 className="text-xs font-semibold text-blue-600 mb-2 px-2">CHAT HISTORY</h3>
             )}
             <div className="space-y-1">
               {chatSessions.slice().reverse().map((session) => (
                 <div key={session.id} className="group relative">
                   <button
                     onClick={() => switchToSession(session.id)}
                     className={`w-full text-left p-2 rounded-lg text-sm transition-colors ${
                       currentSessionId === session.id
                         ? 'bg-blue-100 text-blue-800'
                         : 'text-blue-600 hover:bg-blue-50'
                     } ${isSidebarCollapsed ? 'flex items-center justify-center' : ''}`}
                     title={isSidebarCollapsed ? session.title : undefined}
                   >
                     {isSidebarCollapsed ? (
                       <span className="text-lg">💬</span>
                     ) : (
                       <>
                         <div className="truncate font-medium">{session.title}</div>
                         <div className="text-xs text-blue-400 mt-1">
                           {session.updatedAt.toLocaleDateString()}
                         </div>
                       </>
                     )}
                   </button>
                   {!isSidebarCollapsed && (
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
                   )}
                 </div>
               ))}
               {chatSessions.length === 0 && !isSidebarCollapsed && (
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
                <option value="claude-sonnet-4-5-20250929">🚀 Claude 4.5 Sonnet</option>
                <option value="claude-haiku-4-5-20251001">⚡ Claude 3 Haiku (Fast)</option>
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
        <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-7xl mx-auto">
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
                  className={`${message.type === 'user' ? 'max-w-2xl' : 'w-full max-w-none'} px-8 py-6 rounded-2xl shadow-lg ${
                    message.type === 'user'
                      ? 'bg-blue-100 border border-blue-300 text-gray-900 font-medium'
                      : 'bg-white/90 backdrop-blur-md border border-blue-200/50'
                  }`}
                >
                  {message.isLoading ? (
                    <div className="flex items-center space-x-3 bg-gradient-to-r from-blue-50/80 to-blue-100/60 border border-blue-300/60 rounded-xl p-4 shadow-sm">
                      <div className="relative">
                        <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-200 border-t-blue-600"></div>
                        <div className="absolute inset-0 rounded-full border-2 border-blue-100"></div>
                      </div>
                      <div className="flex-1">
                        {/* Show detailed tool execution if available */}
                        {toolExecutionDetails ? (
                          <div>
                            <div className="text-sm font-semibold text-blue-900">
                              {toolExecutionDetails.currentTool ? `Using ${formatToolName(toolExecutionDetails.currentTool)}` : 'Executing Tools'}
                            </div>
                            <div className="text-xs text-blue-700 mt-1 flex items-center space-x-2">
                              <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span>
                              <span>
                                {toolExecutionDetails.total > 1 
                                  ? `Tool ${toolExecutionDetails.completed + 1} of ${toolExecutionDetails.total}`
                                  : 'Processing with AI tools'
                                }
                              </span>
                            </div>
                            {/* Show tool list if multiple tools */}
                            {toolExecutionDetails.tools.length > 1 && (
                              <div className="text-xs text-blue-600 mt-1">
                                Tools: {toolExecutionDetails.tools.map(formatToolName).join(', ')}
                              </div>
                            )}
                          </div>
                        ) : currentToolStatus ? (
                          <div>
                            <div className="text-sm font-semibold text-blue-900">
                              {currentToolStatus}
                            </div>
                            <div className="text-xs text-blue-700 mt-1 flex items-center space-x-1">
                              <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span>
                              <span>AI is working with tools to provide the best response</span>
                            </div>
                          </div>
                        ) : (
                          <div>
                            <div className="text-sm font-semibold text-blue-900">
                              Processing your request...
                            </div>
                            <div className="text-xs text-blue-600 mt-1">
                              Preparing response...
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="text-xs text-blue-500 font-medium">
                        {toolExecutionDetails || currentToolStatus ? 'Active' : 'Starting'}
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="rfp-analysis-content">
                        <ReactMarkdown 
                          remarkPlugins={[remarkGfm]}
                          className="prose prose-lg max-w-none rfp-markdown"
                          components={{
                            a: ({href, children}) => <a href={href as string} target="_blank" rel="noreferrer" className="underline text-blue-700 hover:text-blue-900">{children}</a>,
                            h1: ({children}) => <h1 className="text-3xl font-bold text-gray-900 mb-6 mt-8 first:mt-0 border-b-2 border-blue-200 pb-3">{children}</h1>,
                            h2: ({children}) => <h2 className="text-2xl font-bold text-blue-800 mb-4 mt-6 first:mt-0 flex items-center gap-2"><span className="text-blue-600">📋</span>{children}</h2>,
                            h3: ({children}) => <h3 className="text-xl font-semibold text-gray-800 mb-3 mt-5 first:mt-0 flex items-center gap-2"><span className="text-gray-600">📄</span>{children}</h3>,
                            h4: ({children}) => <h4 className="text-lg font-bold text-blue-700 mb-3 mt-4 first:mt-0 bg-blue-50/50 px-3 py-2 rounded-lg border-l-4 border-blue-500">{children}</h4>,
                            p: ({children}) => <p className="text-gray-700 leading-relaxed mb-4 text-base">{children}</p>,
                            ul: ({children}) => <ul className="list-disc list-inside space-y-2 mb-4 text-gray-700 ml-2">{children}</ul>,
                            ol: ({children}) => <ol className="list-decimal list-inside space-y-2 mb-4 text-gray-700 ml-2">{children}</ol>,
                            li: ({children}) => <li className="leading-relaxed ml-2">{children}</li>,
                            strong: ({children}) => <strong className="font-extrabold text-blue-900">{children}</strong>,
                            em: ({children}) => <em className="italic text-gray-600">{children}</em>,
                            blockquote: ({children}) => <blockquote className="border-l-4 border-blue-400 pl-4 py-3 bg-blue-50 rounded-r-lg mb-4 italic text-gray-700 font-medium">{children}</blockquote>,
                            code: ({children}) => <code className="bg-gray-100 px-2 py-1 rounded text-sm font-mono text-gray-800">{children}</code>,
                            pre: ({children}) => <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4">{children}</pre>,
                            table: ({children}) => <div className="overflow-x-auto mb-6 shadow-lg rounded-lg"><table className="min-w-full border-collapse border border-gray-300 bg-white">{children}</table></div>,
                            th: ({children}) => <th className="border border-gray-300 bg-blue-700 px-4 py-3 text-left font-bold text-white">{children}</th>,
                            td: ({children}) => <td className="border border-gray-300 px-4 py-3 text-gray-700 bg-white">{children}</td>,
                            hr: () => <hr className="my-8 border-t-2 border-blue-300" />
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                        
                        {/* Import to Dashboard button for RFP analysis */}
                        {message.type === 'assistant' && hasRFPDocumentsTable(message.content) && (() => {
                          // Get the previous user message to extract RFP path
                          const messageIndex = messages.findIndex(m => m.id === message.id)
                          const previousMessage = messageIndex > 0 ? messages[messageIndex - 1] : null
                          const combinedContent = (previousMessage?.content || '') + '\n' + message.content
                          
                          const rfpData = parseRFPDocuments(combinedContent)
                          if (rfpData.hasDocumentsTable && rfpData.documents.length > 0 &&
                                                        rfpData.decision &&
                                                        (rfpData.decision === 'GO' || rfpData.decision === 'CONDITIONAL-GO')) {
                             return (
                              <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-blue-100 border-2 border-blue-300 rounded-xl">
                                <div className="flex items-center justify-between">
                                  <div>
                                    <h4 className="font-bold text-blue-900 text-lg mb-1">📋 Found {rfpData.documents.length} Required Documents</h4>
                                    <p className="text-sm text-blue-700">
                                      Import these as tasks to track document creation progress
                                    </p>
                                  </div>
                                  <button
                                    onClick={async () => {
                                      toast.loading('Preparing import...', { id: 'import-docs' });
                                      try {
                                        // 1) Extract path + folder name from the parsed RFP block
                                        const rfpPath = rfpData.rfpPath || '';
                                        if (!rfpPath) {
                                          toast.error('Could not detect RFP path from analysis', { id: 'import-docs' });
                                          return;
                                        }
                                        const folderName = (rfpPath.split('/').pop() || rfpPath).trim();


                                        // 2) Ensure parent exists (find OR create) and get its id
                                        const parentTaskId = await ensureParentRfpTask(rfpPath, folderName);

                                        // 3) Import the document rows under that parent
                                        toast.loading(`Importing ${rfpData.documents.length} documents...`, { id: 'import-docs' });
                                        const result = await apiClient.importRFPDocuments(
                                          parentTaskId,
                                          rfpPath,
                                          rfpData.documents
                                        );

                                        toast.success(
                                          `✅ Imported ${result.created_tasks?.length ?? rfpData.documents.length} document tasks under “${folderName}”`,
                                          { id: 'import-docs', duration: 5000 }
                                        );

                                        // Optional: jump to dashboard
                                        if (confirm('Documents imported! Open dashboard now?')) {
                                          window.location.href = '/dashboard';
                                        }
                                      } catch (err: any) {
                                        console.error('Import failed:', err);
                                        toast.error(`Failed to import: ${err?.message || err}`, { id: 'import-docs' });
                                      }
                                    }}

                                    className="px-6 py-3 bg-gradient-to-r from-green-600 to-green-700 text-white rounded-xl hover:from-green-700 hover:to-green-800 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 font-semibold flex items-center space-x-2"
                                  >
                                    <span>📋</span>
                                    <span>Import to Dashboard</span>
                                  </button>
                                </div>
                              </div>
                            )
                          }
                          return null
                        })()}
                      </div>
                      {/* Token usage display */}
                      {message.tokenUsage && (
                        <TokenDisplay tokenUsage={message.tokenUsage} compact={true} />
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
                <div className="absolute bottom-full left-0 mb-2 w-[500px] bg-gradient-to-br from-white via-blue-50/50 to-blue-100/30 backdrop-blur-xl border-2 border-blue-200/60 rounded-2xl shadow-2xl z-50 overflow-hidden">
                  <div className="p-6">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-5 pb-3 border-b border-blue-200/50">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-800 rounded-xl flex items-center justify-center shadow-lg">
                          <span className="text-white text-lg">📝</span>
                        </div>
                        <div>
                          <h3 className="text-lg font-bold text-blue-900">Prompt Templates</h3>
                          <p className="text-xs text-blue-600">{allPromptTemplates.length} templates available</p>
                        </div>
                      </div>
                      <button
                        onClick={() => setShowPromptsPanel(false)}
                        className="w-8 h-8 rounded-xl bg-red-100 hover:bg-red-200 text-red-600 flex items-center justify-center transition-all duration-200 hover:scale-105"
                      >
                        ×
                      </button>
                    </div>
                    
                    {/* Templates List */}
                    <div className="space-y-3 max-h-80 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-blue-300 scrollbar-track-blue-100">
                      {allPromptTemplates.length > 0 ? (
                        allPromptTemplates.map((prompt, index) => (
                          <div key={index} className="block">
                            <button
                              onClick={() => handlePromptClick(prompt)}
                              className="w-full text-left p-4 rounded-xl border-2 border-blue-200/60 bg-white/60 hover:border-blue-400 hover:bg-blue-50/50 hover:shadow-md transition-all duration-300 transform hover:scale-[1.02]"
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex-1 flex items-center space-x-3">
                                  <div className="flex items-center space-x-2">
                                    <p className="text-base font-semibold text-blue-900">{prompt.name}</p>
                                    
                                    {/* Add indicator for Step 3 */}
                                    {(prompt.name === 'Go_No_Go_Recommendation' || prompt.name === 'Step3_go_no_go_recommendation') && (
                                      <span className="px-2 py-0.5 bg-orange-100 text-orange-700 text-xs font-medium rounded-full">
                                        📊 Go/No-Go
                                      </span>
                                    )}
                                  </div>
                                </div>
                                <div className="relative">
                                  <button
                                    type="button"
                                    className="group w-7 h-7 rounded-xl bg-gradient-to-br from-blue-100 to-blue-200 text-blue-700 flex items-center justify-center text-sm hover:from-blue-200 hover:to-blue-300 transition-all duration-200 shadow-sm hover:shadow-md transform hover:scale-110"
                                    title={prompt.description}
                                    onClick={(e) => {
                                      e.stopPropagation() // Prevent triggering the parent button
                                    }}
                                  >
                                    ℹ️
                                    <div className="absolute right-8 top-1/2 transform -translate-y-1/2 w-72 p-3 bg-gradient-to-br from-gray-800 to-gray-900 text-white text-sm rounded-xl shadow-2xl opacity-0 group-hover:opacity-100 transition-all duration-300 pointer-events-none z-[60] border border-gray-700">
                                      <div className="font-medium text-blue-300 mb-1">{prompt.name}</div>
                                      <div className="text-gray-200 leading-relaxed">{prompt.description}</div>
                                    </div>
                                  </button>
                                </div>
                              </div>
                            </button>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-8 text-blue-600">
                          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                            <span className="text-2xl">📝</span>
                          </div>
                          <p className="text-base font-medium">No prompt templates available</p>
                          <p className="text-sm mt-1 text-blue-500">Check server connection</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>

                         {/* Folder Selection Modal - Compact Design */}
             {showFolderSelection && (
               <div className="fixed inset-0 bg-black/30 flex items-end justify-center z-50">
                 <div className="bg-white border-t-2 border-blue-500 rounded-t-xl shadow-2xl w-full max-w-2xl max-h-[70vh] flex flex-col">
                   {/* Header */}
                   <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-blue-50 rounded-t-xl">
                     <div className="flex items-center space-x-2">
                       <span className="text-blue-600 text-lg">📁</span>
                       <h3 className="font-semibold text-gray-800">Select SharePoint Folder</h3>
                     </div>
                     <div className="flex items-center space-x-2">
                       <button
                         onClick={() => fetchSharePointFolders(true)}
                         disabled={isLoadingFolders}
                         className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
                         title="Refresh folders"
                       >
                         🔄
                       </button>
                       <button
                         onClick={() => {
                           setShowFolderSelection(false)
                           setSelectedPrompt(null)
                         }}
                         className="w-6 h-6 rounded-full bg-gray-500 text-white text-xs flex items-center justify-center hover:bg-gray-600 transition-colors"
                       >
                         ×
                       </button>
                     </div>
                   </div>
                   
                   {/* Content */}
                   <div className="flex-1 overflow-hidden flex flex-col">
                     <div className="p-4 border-b border-gray-100">
                       <p className="text-sm text-gray-600">
                         Choose the SharePoint folder you want to 
                         {selectedPrompt && (selectedPrompt.name === 'Step1_Identifying_documents' || 
                          selectedPrompt.name === 'Step-1: Document Identification Assistant')
                           ? ' categorize primary RFP documents from:'
                           : ' analyze for RFP/RFI/RFQ documents:'}
                       </p>
                       
                       {/* Show special message for Step 1 */}
                       {selectedPrompt && (selectedPrompt.name === 'Step1_Identifying_documents' || 
                         selectedPrompt.name === 'Step-1: Document Identification Assistant') && (
                         <div className="bg-green-50 border border-green-200 rounded-lg p-2 mt-2">
                           <div className="flex items-start space-x-2">
                             <span className="text-green-600 text-sm">ℹ️</span>
                             <div className="text-xs text-green-700">
                               <strong>Primary RFP Categorization:</strong> Step 1 will categorize whether each document is a primary RFP document or not.
                             </div>
                           </div>
                         </div>
                       )}
                     </div>
                     
                     {/* Folder List */}
                     <div className="flex-1 overflow-y-auto p-4">
                       <div className="grid grid-cols-1 gap-2">
                         {availableFolders.length > 0 ? (
                           availableFolders.map((folderName, index) => (
                             <button
                               key={index}
                               onClick={() => handleFolderSelection(folderName)}
                               className="w-full text-left p-3 bg-gray-50 hover:bg-blue-50 rounded-lg transition-colors border border-gray-200 hover:border-blue-300 group"
                             >
                               <div className="flex items-center space-x-3">
                                 <span className="text-gray-500 group-hover:text-blue-600 transition-colors">📁</span>
                                 <span className="font-medium text-gray-800 group-hover:text-blue-800">{folderName}</span>
                               </div>
                             </button>
                           ))
                         ) : (
                           <div className="text-center py-8 text-gray-500">
                             <div className="text-4xl mb-2">📂</div>
                             <p className="text-sm">{isLoadingFolders ? 'Loading folders...' : 'No folders found'}</p>
                             {isLoadingFolders && (
                               <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mx-auto mt-2"></div>
                             )}
                           </div>
                         )}
                       </div>
                     </div>
                     
                     {/* Footer */}
                     <div className="p-4 border-t border-gray-200 bg-gray-50">
                       <button
                         onClick={() => {
                           setShowFolderSelection(false)
                           setSelectedPrompt(null)
                         }}
                         className="w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                       >
                         Cancel
                       </button>
                     </div>
                   </div>
                 </div>
               </div>
             )}

             {/* Subfolder Selection Modal - Compact Design */}
             {showSubFolderSelection && (
               <div className="fixed inset-0 bg-black/30 flex items-end justify-center z-50">
                 <div className="bg-white border-t-2 border-blue-500 rounded-t-xl shadow-2xl w-full max-w-2xl max-h-[70vh] flex flex-col">
                   {/* Header */}
                   <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-blue-50 rounded-t-xl">
                     <div className="flex items-center space-x-2">
                       <span className="text-blue-600 text-lg">📁</span>
                       <h3 className="font-semibold text-gray-800">Select Project Folder</h3>
                     </div>
                     <div className="flex items-center space-x-2">
                       <button
                          onClick={async () => {
                            await fetchSubFolders(selectedParentFolder, true);
                            const step2 =
                              (selectedPrompt?.name || '').toLowerCase().includes('summarize') ||
                              (selectedPrompt?.name || '').toLowerCase().includes('step2') ||
                              (selectedPrompt?.description || '').toLowerCase().includes('generate a clear, high-value summary');
                            if (step2 && selectedParentFolder) {
                              await fetchSharePointFiles(selectedParentFolder, true);
                            }
                          }}
                          disabled={isLoadingFolders}
                          className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
                          title="Refresh"
                        >
                          🔄
                        </button>

                       <button
                         onClick={() => {
                           setShowSubFolderSelection(false)
                           setSelectedPrompt(null)
                           setSelectedParentFolder('')
                         }}
                         className="w-6 h-6 rounded-full bg-gray-500 text-white text-xs flex items-center justify-center hover:bg-gray-600 transition-colors"
                       >
                         ×
                       </button>
                     </div>
                   </div>
                   
                   {/* Content */}
                   <div className="flex-1 overflow-hidden flex flex-col">
                     <div className="p-4 border-b border-gray-100">
                       <p className="text-sm text-gray-600">
                         Choose a project folder within <span className="font-medium text-blue-600">{selectedParentFolder}</span> to browse documents:
                       </p>
                     </div>
                   
                     {/* Subfolder List */}
                     <div className="flex-1 overflow-y-auto p-4">
                       <div className="grid grid-cols-1 gap-2">
                         {subFolders.length > 0 ? (
                           subFolders.map((folderName, index) => (
                             <button
                               key={index}
                               onClick={() => handleSubFolderSelection(folderName)}
                               className="w-full text-left p-3 bg-gray-50 hover:bg-blue-50 rounded-lg transition-colors border border-gray-200 hover:border-blue-300 group"
                             >
                               <div className="flex items-center space-x-3">
                                 <span className="text-gray-500 group-hover:text-blue-600 transition-colors">📁</span>
                                 <span className="font-medium text-gray-800 group-hover:text-blue-800">{folderName}</span>
                               </div>
                             </button>
                           ))
                         ) : (
                           <div className="text-center py-8 text-gray-500">
                             <div className="text-4xl mb-2">📂</div>
                             <p className="text-sm">{isLoadingFolders ? 'Loading subfolders...' : 'No subfolders found'}</p>
                             {isLoadingFolders && (
                               <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mx-auto mt-2"></div>
                             )}
                           </div>
                         )}
                       </div>
                     </div>
                     {/* ——— Files in this folder (Summarize / Step 2 only) ——— */}
                      {(() => {
                        const step2 =
                          (selectedPrompt?.name || '').toLowerCase().includes('summarize') ||
                          (selectedPrompt?.name || '').toLowerCase().includes('step2') ||
                          (selectedPrompt?.description || '').toLowerCase().includes('generate a clear, high-value summary');

                        if (!step2) return null;

                        return (
                          <div className="px-4 pb-4">
                            <div className="text-sm font-semibold text-gray-700 mb-2">
                              Files in “{selectedParentFolder}”
                            </div>

                            {isLoadingFiles ? (
                              <div className="text-center py-6 text-gray-500">
                                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mx-auto mb-2"></div>
                                Loading files…
                              </div>
                            ) : availableFiles.length > 0 ? (
                              <div className="space-y-2 max-h-56 overflow-y-auto">
                                {availableFiles.map((file, idx) => (
                                  <button
                                    key={`${file.path}-${idx}`}
                                    onClick={async () => {
                                      await handleFileSelection(file);  // uses your existing handler
                                      setShowSubFolderSelection(false); // close modal after pick
                                      setSelectedParentFolder('');
                                    }}
                                    className="w-full text-left p-3 bg-blue-50/40 hover:bg-blue-100 rounded-lg transition-colors border border-blue-200/60"
                                  >
                                    <div className="flex items-center justify-between">
                                      <div className="flex items-center gap-3">
                                        <span className="text-blue-600">
                                          {file.extension === 'pdf'
                                            ? '📄'
                                            : file.extension === 'docx'
                                            ? '📝'
                                            : file.extension === 'xlsx'
                                            ? '📊'
                                            : '📄'}
                                        </span>
                                        <div>
                                          <div className="font-medium text-gray-800">{file.name}</div>
                                          <div className="text-xs text-blue-700 mt-1">
                                            {file.size_mb ? `${file.size_mb} MB` : 'Unknown size'}
                                            {file.extension ? ` • ${String(file.extension).toUpperCase()}` : ''}
                                            {file.modified_date
                                              ? ` • Modified: ${new Date(file.modified_date).toLocaleDateString()}`
                                              : ''}
                                          </div>
                                        </div>
                                      </div>
                                      <div className="text-blue-700 text-sm">Summarize →</div>
                                    </div>
                                  </button>
                                ))}
                              </div>
                            ) : (
                              <div className="text-center py-6 text-gray-500">No files in this folder.</div>
                            )}
                          </div>
                        );
                      })()}

                     {/* Footer */}
                     <div className="p-4 border-t border-gray-200 bg-gray-50 flex space-x-2">
                       <button
                         onClick={() => {
                           setShowSubFolderSelection(false)
                           setShowFolderSelection(true)
                           setSelectedParentFolder('')
                         }}
                         className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                       >
                         ← Back
                       </button>
                       <button
                         onClick={() => {
                           setShowSubFolderSelection(false)
                           setSelectedPrompt(null)
                           setSelectedParentFolder('')
                         }}
                         className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                       >
                         Cancel
                       </button>
                     </div>
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
                          selectedPrompt.name === 'Step-1: Document Identification Assistant')
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
                        selectedPrompt.name === 'Step-1: Document Identification Assistant')
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
                                 selectedPrompt.name === 'Step-1: Document Identification Assistant')
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

            {/* Document Type Input Modal for Step 4 */}
            {showDocumentTypeInput && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                <div className="bg-white/95 backdrop-blur-md border border-blue-100/50 rounded-xl shadow-xl p-6 max-w-md w-full mx-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-bold text-blue-800 text-lg">📄 Generate Document</h3>
                    <button
                      onClick={handleDocumentTypeCancel}
                      className="w-6 h-6 rounded-full bg-gray-500 text-white text-xs flex items-center justify-center hover:bg-gray-600 transition-colors"
                    >
                      ×
                    </button>
                  </div>
                  
                  <div className="mb-4">
                    <p className="text-sm text-gray-600 mb-3">
                      Which document would you like to generate? The system will use your previous RFP analysis and BroadAxis knowledge base.
                    </p>
                    
                    <div className="space-y-2 mb-4">
                      <p className="text-xs text-gray-500 font-medium">Common document types:</p>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded">• Capability Statement</span>
                        <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded">• Statement of Work</span>
                        <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded">• Technical Proposal</span>
                        <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded">• Project Plan</span>
                        <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded">• Risk Management</span>
                        <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded">• Cost Proposal</span>
                      </div>
                    </div>
                    
                    <input
                      type="text"
                      value={documentTypeInput}
                      onChange={(e) => setDocumentTypeInput(e.target.value)}
                      placeholder="e.g., Statement of Work, Technical Proposal, etc."
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      onKeyPress={(e) => e.key === 'Enter' && handleDocumentTypeSubmit()}
                      autoFocus
                    />
                  </div>
                  
                  <div className="flex space-x-2">
                    <button
                      onClick={handleDocumentTypeSubmit}
                      disabled={!documentTypeInput.trim()}
                      className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Generate Document
                    </button>
                    <button
                      onClick={handleDocumentTypeCancel}
                      className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
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
                <div className="absolute bottom-full left-0 mb-2 w-[500px] bg-gradient-to-br from-white via-blue-50/50 to-blue-100/30 backdrop-blur-xl border-2 border-blue-200/60 rounded-2xl shadow-2xl z-50 overflow-hidden">
                  <div className="p-6">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-5 pb-3 border-b border-blue-200/50">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-800 rounded-xl flex items-center justify-center shadow-lg">
                          <span className="text-white text-lg">🔧</span>
                        </div>
                        <div>
                          <h3 className="text-lg font-bold text-blue-900">Available Tools</h3>
                          <p className="text-xs text-blue-600">{availableTools.length} tools • {settings.enabledTools.length} enabled</p>
                        </div>
                      </div>
                      <button
                        onClick={() => setShowToolsPanel(false)}
                        className="w-8 h-8 rounded-xl bg-red-100 hover:bg-red-200 text-red-600 flex items-center justify-center transition-all duration-200 hover:scale-105"
                      >
                        ×
                      </button>
                    </div>
                    
                    {/* Action Buttons */}
                    <div className="flex space-x-3 mb-5">
                      <button
                        onClick={() => {
                          const allToolNames = availableTools.map(tool => tool.name)
                          setSettings(prev => ({ ...prev, enabledTools: allToolNames }))
                          localStorage.setItem('broadaxis-enabled-tools', JSON.stringify(allToolNames))
                        }}
                        className="flex-1 px-4 py-2 text-sm font-medium bg-gradient-to-r from-green-600 to-green-700 text-white rounded-xl hover:from-green-700 hover:to-green-800 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
                      >
                        ✅ Enable All
                      </button>
                      <button
                        onClick={() => {
                          setSettings(prev => ({ ...prev, enabledTools: [] }))
                          localStorage.setItem('broadaxis-enabled-tools', JSON.stringify([]))
                        }}
                        className="flex-1 px-4 py-2 text-sm font-medium bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-xl hover:from-orange-600 hover:to-orange-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
                      >
                        🚫 Disable All
                      </button>
                    </div>
                    
                    {/* Tools List */}
                    <div className="space-y-3 max-h-80 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-blue-300 scrollbar-track-blue-100">
                      {availableTools.map((tool) => (
                        <div key={tool.name} className="block">
                          <div className={`p-4 rounded-xl border-2 transition-all duration-300 transform hover:scale-[1.02] ${
                            settings.enabledTools.includes(tool.name)
                              ? 'border-blue-500 bg-gradient-to-r from-blue-50 to-blue-100/50 shadow-lg'
                              : 'border-blue-200/60 bg-white/60 hover:border-blue-400 hover:bg-blue-50/50 hover:shadow-md'
                          }`}>
                            <div className="flex items-center space-x-3">
                              <div className="relative">
                                <input
                                  type="checkbox"
                                  checked={settings.enabledTools.includes(tool.name)}
                                  onChange={() => toggleTool(tool.name)}
                                  className="w-5 h-5 rounded-md border-2 border-blue-300 text-blue-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 cursor-pointer"
                                />
                                {settings.enabledTools.includes(tool.name) && (
                                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full flex items-center justify-center">
                                    <span className="text-white text-xs">✓</span>
                                  </div>
                                )}
                              </div>
                              <div className="flex-1 flex items-center justify-between">
                                <div className="flex items-center space-x-2">
                                  <p className="text-base font-semibold text-blue-900">{tool.name}</p>
                                  {settings.enabledTools.includes(tool.name) && (
                                    <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs font-medium rounded-full">
                                      Active
                                    </span>
                                  )}
                                </div>
                                <div className="relative">
                                  <button
                                    type="button"
                                    className="group w-7 h-7 rounded-xl bg-gradient-to-br from-blue-100 to-blue-200 text-blue-700 flex items-center justify-center text-sm hover:from-blue-200 hover:to-blue-300 transition-all duration-200 shadow-sm hover:shadow-md transform hover:scale-110"
                                    title={tool.description}
                                  >
                                    ℹ️
                                    <div className="absolute right-8 top-1/2 transform -translate-y-1/2 w-72 p-3 bg-gradient-to-br from-gray-800 to-gray-900 text-white text-sm rounded-xl shadow-2xl opacity-0 group-hover:opacity-100 transition-all duration-300 pointer-events-none z-[60] border border-gray-700">
                                      <div className="font-medium text-blue-300 mb-1">{tool.name}</div>
                                      <div className="text-gray-200 leading-relaxed">{tool.description}</div>
                                    </div>
                                  </button>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Message Input */}
            <textarea
              ref={setTextareaRef}
              value={inputMessage}
              onChange={handleInputChange}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSendMessage()
                }
              }}
              placeholder={uploadedFiles.length > 0 ? "Ask questions about your uploaded documents..." : "Message BroadAxis-AI..."}
              className="flex-1 px-6 py-4 border border-blue-200/50 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white/80 backdrop-blur-md shadow-lg placeholder-blue-500 resize-none min-h-[3rem] max-h-[12rem] overflow-y-auto"
              disabled={isLoading}
              rows={1}
              style={{ height: 'auto' }}
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
                   setCurrentToolStatus('')
                   setToolExecutionDetails(null)
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
               ↑
             </button>
          </div>
          

        </div>
      </div>

      {/* Token Dashboard Modal */}
      {showTokenDashboard && (
        <TokenDashboard
          sessionId={currentSessionId || 'default'}
          userId="ecf91d9c-7041-4b20-805f-768b6d8d5ec1" // TODO: Get from auth context
          onClose={() => setShowTokenDashboard(false)}
        />
      )}

    </div>
  )
}

export default ChatInterface
