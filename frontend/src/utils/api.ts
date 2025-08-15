import axios from 'axios'
import { ChatRequest, ChatResponse, FileInfo, Tool, Prompt, UploadResponse, GeneratedFile, EmailFetchRequest, EmailFetchResponse } from '../types'

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 5000, // 5 seconds default timeout
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
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
    
    // Handle specific error cases
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout')
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

  // Chat endpoints
  async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await api.post('/api/chat', request, {
      timeout: 120000 // 2 minutes for chat messages
    })
    return response.data
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

  // Document chat with hybrid retrieval
  async chatWithDocument(query: string, sessionId: string = 'default', filename?: string): Promise<ChatResponse> {
    const response = await api.post('/api/chat/document', {
      query,
      enabled_tools: [],
      model: 'claude-3-haiku-20240307'
    }, {
      params: { session_id: sessionId, filename },
      timeout: 60000
    })
    return response.data
  },

  // Session file management
  async getSessionFiles(sessionId: string): Promise<{ files: FileInfo[]; total_chunks?: number }> {
    try {
      const response = await api.get(`/api/files/${sessionId}`)
      return response.data
    } catch (error) {
      console.error('Failed to get session files:', error)
      return { files: [] }
    }
  },

  async clearSession(sessionId: string): Promise<{ message: string }> {
    const response = await api.delete(`/api/files/${sessionId}`)
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
          { name: "Broadaxis_knowledge_search", description: "Search company knowledge base", input_schema: {} },
          { name: "web_search_tool", description: "Search the web", input_schema: {} },
          { name: "generate_pdf_document", description: "Generate PDF documents", input_schema: {} },
          { name: "generate_word_document", description: "Generate Word documents", input_schema: {} },
          { name: "generate_text_file", description: "Generate text files", input_schema: {} },
          { name: "search_papers", description: "Search academic papers", input_schema: {} },
          { name: "get_forecast", description: "Get weather forecast", input_schema: {} },
          { name: "get_alerts", description: "Get weather alerts", input_schema: {} }
        ],
        prompts: [
          { name: "Step-2: Executive Summary", description: "Generate executive summary of RFP documents", arguments: [] },
          { name: "Step-3: Go/No-Go Recommendation", description: "Provide Go/No-Go analysis", arguments: [] },
          { name: "Step-4: Generate Proposal", description: "Generate capability statement", arguments: [] }
        ],
        status: "fallback"
      }
    }
  },

  // Tools and prompts (legacy endpoints - now use cached data)
  async getAvailableTools(): Promise<{ tools: Tool[]; status: string }> {
    try {
      const response = await api.get('/api/tools')
      return response.data
    } catch (error) {
      console.error('Failed to get tools:', error)
      return { tools: [], status: "error" }
    }
  },

  async getAvailablePrompts(): Promise<{ prompts: Prompt[]; status: string }> {
    try {
      const response = await api.get('/api/prompts')
      return response.data
    } catch (error) {
      console.error('Failed to get prompts:', error)
      return { prompts: [], status: "error" }
    }
  },



  // File management endpoints
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
      }, { timeout: 30000 }) // Reduced to 30 seconds
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

  // Test Microsoft Graph API authentication
  async testGraphAuth(): Promise<{ status: string; message: string; step?: string }> {
    try {
      const response = await api.get('/api/test-graph-auth', { timeout: 15000 })
      return response.data
    } catch (error) {
      console.error('Failed to test Graph API auth:', error)
      return {
        status: 'error',
        message: 'Failed to test Microsoft Graph API authentication',
        step: 'network_error'
      }
    }
  },


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