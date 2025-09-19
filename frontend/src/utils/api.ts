import axios from 'axios'
import { ChatResponse, FileInfo, Tool, Prompt, UploadResponse, GeneratedFile, EmailFetchRequest, EmailFetchResponse } from '../types'

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000'

// Ensures "A/B/C" becomes "A/B/C" with each piece safely encoded


// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // Increased to 30 seconds for complex operations
  headers: {
    'Content-Type': 'application/json',
  },
})

const encodePath = (p: string) =>
  (p || "").split("/").map(encodeURIComponent).join("/");

// Request interceptor for logging and auth
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    
    // Add auth token to requests
    const token = localStorage.getItem('access_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for auth error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid, clear it
      localStorage.removeItem('access_token')
      console.warn('Authentication token expired or invalid')
      // Optionally redirect to login or show auth error
    }
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message)
    
    // Enhanced error handling
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout - consider reducing complexity or waiting')
    } else if (error.code === 'ECONNREFUSED') {
      console.error('Connection refused - server may be down')
    } else if (!error.response) {
      console.error('Network error - no response received')
    }
    
    return Promise.reject(error)
  }
)

// API functions
export const apiClient = {
  // Health check
  async healthCheck() {
    try {
      const response = await api.get('/health')
      return response.data
    } catch (error) {
      console.error('Health check failed:', error)
      return {
        status: "error",
        message: "Server not responding"
      }
    }
  },

  // File upload endpoints
  async uploadFile(file: File, sessionId: string = 'default'): Promise<FileInfo> {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await api.post(`/api/upload?session_id=${sessionId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000 // 60 seconds for file upload
    })
    return response.data
  },

  // Initialize MCP server (single call for tools and prompts)
  async initializeMCP(): Promise<{ tools: Tool[]; prompts: Prompt[]; status: string }> {
    try {
      const response = await api.post('/api/initialize', {}, { timeout: 120000 })
      return response.data
    } catch (error) {
      console.error('Failed to initialize MCP:', error)
      return {
        tools: [
          { name: "sum", description: "Add two numbers", input_schema: {} },
          { name: "Broadaxis_knowledge_search", description: "Search company knowledge base with semantic search over Pinecone using OpenAI embeddings", input_schema: {} },
          { name: "web_search_tool", description: "Perform real-time web search using Tavily API", input_schema: {} },
          { name: "generate_pdf_document", description: "Generate and create professional PDF documents with markdown support and automatic SharePoint upload", input_schema: {} },
          { name: "generate_word_document", description: "Generate and create professional Word documents with enhanced markdown support and automatic SharePoint upload", input_schema: {} },
          { name: "sharepoint_read_file", description: "Read files from SharePoint with enhanced features and validation", input_schema: {} },
          { name: "sharepoint_list_files", description: "List files and directories in SharePoint folder with enhanced filtering and sorting", input_schema: {} },
          { name: "sharepoint_search_files", description: "Search for files in SharePoint with enhanced search capabilities", input_schema: {} },
          { name: "extract_pdf_text", description: "Extract and process text content from PDF files in SharePoint with enhanced features for RFP analysis", input_schema: {} }
        ],
        prompts: [
          { name: "Summarize_Document", description: "Generate a clear, high-value summary of SharePoint RFP, RFQ, or RFI documents for executive decision-making", arguments: [] },
          { name: "Go_No_Go_Recommendation", description: "Evaluate whether BroadAxis should pursue an RFP, RFQ, or RFI opportunity with structured Go/No-Go analysis", arguments: [] },
          { name: "Dynamic_Content_Generator", description: "Dynamic Document Generator - Create various professional documents for RFP/RFQ responses", arguments: [] },
          { name: "Fill_Missing_Information", description: "Fill in missing fields and answer RFP/RFQ questions using verified information from internal knowledge base", arguments: [] }
        ],
        status: "fallback"
      }
    }
  },

  // File management endpoints (SharePoint)
  async listFiles(): Promise<{ files: any[]; status: string; message: string }> {
    try {
      const response = await api.get('/api/files', { timeout: 15000 })
      return response.data
    } catch (error) {
      console.error('Failed to list files:', error)
      return {
        files: [],
        status: "error",
        message: "Failed to fetch files"
      }
    }
  },

  // Email fetching endpoints
  async fetchEmails(emailAccounts: string[] = [], useRealEmail: boolean = true, useGraphApi: boolean = true): Promise<EmailFetchResponse> {
    try {
      const response = await api.post('/api/fetch-emails', {
        email_accounts: emailAccounts,
        use_real_email: useRealEmail,
        use_graph_api: useGraphApi
      }, { timeout: 600000})
      return response.data
    } catch (error: any) {
      console.error('Failed to fetch emails:', error)
      if (error.code === 'ECONNABORTED') {
        throw new Error('Email fetch timeout - please check your Microsoft Graph API configuration')
      }
      throw error
    }
  },

  async getFetchedEmails(): Promise<{ emails: any[]; total_count: number; total_files: number }> {
    try {
      const response = await api.get('/api/fetched-emails', { timeout: 15000 })
      return response.data
    } catch (error) {
      console.error('Failed to get fetched emails:', error)
      return {
        emails: [],
        total_count: 0,
        total_files: 0
      }
    }
  },

  async getEmailAttachments(emailId: number): Promise<{ email_id: number; attachments: any[] }> {
    try {
      const response = await api.get(`/api/email-attachments/${emailId}`, { timeout: 15000 })
      return response.data
    } catch (error) {
      console.error('Failed to get email attachments:', error)
      return {
        email_id: emailId,
        attachments: []
      }
    }
  },

  // Test SharePoint connection
  async testSharePoint(): Promise<{ test_result: { status: string; message: string; step?: string } }> {
    try {
      const response = await api.get('/api/test-sharepoint', { timeout: 15000 })
      return response.data
    } catch (error) {
      console.error('Failed to test SharePoint:', error)
      return {
        test_result: {
          status: 'error',
          message: 'Failed to test SharePoint connection',
          step: 'network_error'
        }
      }
    }
  },

  // SharePoint and PDF processing tools - using direct API endpoints
  async listSharePointFiles(path: string = "", fileType?: string, sortBy: string = "name", sortOrder: string = "asc", maxItems: number = 100): Promise<any> {
    try {
      // Use the direct SharePoint API endpoint
      const endpoint = path ? `/api/files/${encodePath(path)}` : '/api/files'
      const response = await api.get(endpoint, { timeout: 60000 })

      return response.data
    } catch (error) {
      console.error('Failed to list SharePoint files:', error)
      throw error
    }
  },

  async readSharePointFile(path: string, maxSizeMb: number = 50, encoding: string = "utf-8", previewLines: number = 0): Promise<any> {
    try {
      const response = await api.get(`/api/files/${encodePath(path)}`, { timeout: 60000 })
      return response.data
    } catch (error) {
      console.error('Failed to read SharePoint file:', error)
      throw error
    }
  },

  async searchSharePointFiles(query: string, path: string = "", searchType: string = "filename", fileType?: string, maxResults: number = 50, includeContent: boolean = false): Promise<any> {
    try {
      // For now, we'll use the list files endpoint and filter client-side
      // In the future, we can add a search endpoint to the backend
      const response = await api.get('/api/files', { timeout: 60000 })
      const files = response.data.files || []
      
      // Simple client-side filtering
      const filteredFiles = files.filter((file: any) => 
        file.name.toLowerCase().includes(query.toLowerCase()) ||
        file.path.toLowerCase().includes(query.toLowerCase())
      )
      
      return {
        files: filteredFiles.slice(0, maxResults),
        total: filteredFiles.length
      }
    } catch (error) {
      console.error('Failed to search SharePoint files:', error)
      throw error
    }
  },

  // Get SharePoint folder counts for dashboard
  async getSharePointFolderCounts(): Promise<{ rfp: number; rfi: number; rfq: number }> {
    try {
      const counts = { rfp: 0, rfi: 0, rfq: 0 }
      
      // Get counts for each folder type
      const folders = ['RFP', 'RFI', 'RFQ']
      
      for (const folder of folders) {
        try {
          console.log(`Fetching count for ${folder} folder...`)
          const response = await api.get(`/api/files/${encodePath(folder)}`, { timeout: 60000 })
          
          console.log(`${folder} response:`, response.data)
          
          if (response.data && response.data.files) {
            // Count files (not folders) in each directory
            const files = response.data.files.filter((item: any) => item.type === 'file')
            const fileCount = files.length
            console.log(`${folder} folder has ${fileCount} files:`, files.map((f: any) => f.name))
            counts[folder.toLowerCase() as keyof typeof counts] = fileCount
          } else {
            console.warn(`${folder} response data:`, response.data)
          }
        } catch (error) {
          console.error(`Failed to get count for ${folder} folder:`, error)
          // Continue with other folders even if one fails
        }
      }
      
      console.log('Final counts:', counts)
      return counts
    } catch (error) {
      console.error('Failed to get SharePoint folder counts:', error)
      return { rfp: 0, rfi: 0, rfq: 0 }
    }
  },

  // Session file management - simplified since we removed session-specific endpoints
  async getSessionFiles(sessionId: string): Promise<{ files: FileInfo[]; total_chunks?: number }> {
    try {
      // For now, return empty array since we removed session-specific file management
      // In the future, we can implement this using the upload endpoint response
      return { files: [] }
    } catch (error) {
      console.error('Failed to get session files:', error)
      return { files: [] }
    }
  },

  async clearSession(sessionId: string): Promise<{ message: string }> {
    // Since we removed session management, just return success
    return { message: "Session cleared (session management disabled)" }
  },

  // Session management endpoints
  async createSession(): Promise<{ session_id: string }> {
    const response = await api.post('/api/session/create')
    return response.data
  },

  // Authentication API methods
  async register(userData: { name: string; email: string; password: string }) {
    const response = await api.post('/api/auth/register', userData)
    return response.data
  },

  async login(userData: { email: string; password: string }) {
    const response = await api.post('/api/auth/login', userData)
    return response.data
  },

  async forgotPassword(payload: { email: string }) {
    const response = await api.post('/api/auth/forgot', payload)
    return response.data as { status: string; reset_token?: string }
  },

  async resetPassword(payload: { token: string; new_password: string }) {
    const response = await api.post('/api/auth/reset', payload)
    return response.data as { status: string }
  },

  async logout() {
    const response = await api.post('/api/auth/logout')
    return response.data
  },

  async getCurrentUser() {
    const response = await api.get('/api/auth/me')
    return response.data
  },

  async getTradingAccess(): Promise<{ trading_access: boolean; status: string }> {
    const response = await api.get('/api/access/trading')
    return response.data
  },

  async getSessionInfo(sessionId: string): Promise<any> {
    const response = await api.get(`/api/session/${sessionId}`)
    return response.data
  },

  async getUserSessions(): Promise<any> {
    const response = await api.get('/api/user/sessions')
    return response.data
  },

  // Trading endpoints
  async tradingCreateSession(): Promise<{ session_id: string }> {
    const response = await api.post('/api/trading/session/create')
    return response.data
  },
  async tradingListSessions(): Promise<any> {
    const response = await api.get('/api/trading/sessions')
    return response.data
  },
  async tradingGetSession(sessionId: string): Promise<any> {
    const response = await api.get(`/api/trading/session/${sessionId}`)
    return response.data
  },
  async tradingDeleteSession(sessionId: string): Promise<any> {
    const response = await api.delete(`/api/trading/session/${sessionId}`)
    return response.data
  },
  async tradingChat(query: string, sessionId?: string, model: string = 'claude-3-7-sonnet-20250219'): Promise<any> {
    const response = await api.post('/api/trading/chat', { query, session_id: sessionId, model })
    return response.data
  },

  async deleteSession(sessionId: string): Promise<any> {
    const response = await api.delete(`/api/session/${sessionId}`)
    return response.data
  },

  async getRedisStatus(): Promise<any> {
    const response = await api.get('/api/redis/status')
    return response.data
  },

  // New intelligent RFP processing
  async processRFPFolderIntelligent(folderPath: string, sessionId: string = 'default'): Promise<any> {
    try {
      const response = await api.post('/api/process-rfp-folder-intelligent', {
        folder_path: folderPath,
        session_id: sessionId
      }, { timeout: 300000 }); // 5 minute timeout
      
      return response.data;
    } catch (error) {
      console.error('Failed to process RFP folder intelligently:', error);
      throw error;
    }
  },

  // Token management endpoints
  async getTokenStatus(): Promise<any> {
    try {
      const response = await api.get('/api/token-status');
      return response.data;
    } catch (error) {
      console.error('Failed to get token status:', error);
      throw error;
    }
  },

  async getTokenUsage(sessionId?: string): Promise<any> {
    try {
      const url = sessionId ? `/api/token-usage/${sessionId}` : '/api/token-usage';
      const response = await api.get(url);
      return response.data;
    } catch (error) {
      console.error('Failed to get token usage:', error);
      throw error;
    }
  },

  // New token tracking methods
  async getQueryTokenUsage(requestId: string): Promise<any> {
    try {
      const response = await api.get(`/api/token-usage/query/${requestId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get query token usage:', error);
      throw error;
    }
  },

  async getUserTokenUsage(userId: string): Promise<any> {
    try {
      const response = await api.get(`/api/token-usage/user/${userId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get user token usage:', error);
      throw error;
    }
  },

  async getDetailedSessionTokenUsage(sessionId: string): Promise<any> {
    try {
      const response = await api.get(`/api/token-usage/detailed/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get detailed session token usage:', error);
      throw error;
    }
  },

  // Folder upload functionality
  async uploadFolder(files: FileList, targetFolder: string, sessionId: string = 'default'): Promise<any> {
    try {
      const formData = new FormData()
      
      // Add files to FormData
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i])
      }
      
      // Add target folder and session ID
      formData.append('target_folder', targetFolder)
      formData.append('session_id', sessionId)
      
      const response = await api.post('/api/upload-folder', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes timeout for folder uploads
      })
      
      return response.data
    } catch (error) {
      console.error('Failed to upload folder:', error)
      throw error
    }
  }

}

// WebSocket utility
export class ChatWebSocket {
  private ws: WebSocket | null = null
  private url: string
  private onMessage: (message: any) => void
  private onError: (error: Event) => void
  private onClose: () => void

  constructor(
    onMessage: (message: any) => void,
    onError: (error: Event) => void = () => {},
    onClose: () => void = () => {}
  ) {
    this.url = `${API_BASE_URL.replace('http', 'ws')}/ws/chat`
    this.onMessage = onMessage
    this.onError = onError
    this.onClose = onClose
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url)
        
        this.ws.onopen = () => {
          console.log('WebSocket connected')
          resolve()
        }
        
        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            this.onMessage(data)
          } catch (error) {
            console.error('Error parsing WebSocket message:', error)
          }
        }
        
        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          this.onError(error)
          reject(error)
        }
        
        this.ws.onclose = () => {
          console.log('WebSocket disconnected')
          this.onClose()
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  sendMessage(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.error('WebSocket is not connected')
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }
}


export default api