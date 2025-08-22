import axios from 'axios'
import { ChatResponse, FileInfo, Tool, Prompt, UploadResponse, GeneratedFile, EmailFetchRequest, EmailFetchResponse } from '../types'

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // Increased to 30 seconds for complex operations
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging and rate limit awareness
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    
    // Add rate limit awareness headers
    config.headers['X-Client-Version'] = '1.0.0'
    config.headers['X-Request-Type'] = config.url?.includes('/chat') ? 'ai-request' : 'standard'
    
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for error handling with rate limit awareness
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message)
    
    // Enhanced error handling for rate limits and timeouts
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout - consider reducing complexity or waiting')
    } else if (error.code === 'ECONNREFUSED') {
      console.error('Connection refused - server may be down')
    } else if (!error.response) {
      console.error('Network error - no response received')
    } else if (error.response?.status === 429) {
      console.error('Rate limit exceeded - server is throttling requests')
      // Could implement exponential backoff here
    } else if (error.response?.status === 503) {
      console.error('Service temporarily unavailable - server overloaded')
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

  // Document chat with hybrid retrieval
  async chatWithDocument(query: string, sessionId: string = 'default', filename?: string): Promise<ChatResponse> {
    const response = await api.post('/api/chat/document', {
      query,
      enabled_tools: [],
      model: 'claude-3-7-sonnet-20250219'
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
          { name: "Step1_Identifying_documents", description: "Browse SharePoint folders to identify and categorize RFP/RFI/RFQ documents from available folders", arguments: [] },
          { name: "Step2_summarize_documents", description: "Generate a clear, high-value summary of SharePoint RFP, RFQ, or RFI documents for executive decision-making", arguments: [] },
          { name: "Step3_go_no_go_recommendation", description: "Evaluate whether BroadAxis should pursue an RFP, RFQ, or RFI opportunity with structured Go/No-Go analysis", arguments: [] },
          { name: "Step4_generate_capability_statement", description: "Generate high-quality capability statements and proposal documents for RFP and RFQ responses", arguments: [] },
          { name: "Step5_fill_missing_information", description: "Fill in missing fields and answer RFP/RFQ questions using verified information from internal knowledge base", arguments: [] }
        ],
        status: "fallback"
      }
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

  // SharePoint and PDF processing tools
  async listSharePointFiles(path: string = "", fileType?: string, sortBy: string = "name", sortOrder: string = "asc", maxItems: number = 100): Promise<any> {
    try {
      // Use the direct SharePoint API endpoint
      const endpoint = path ? `/api/files/${encodeURIComponent(path)}` : '/api/files'
      const response = await api.get(endpoint, { timeout: 60000 })
      return response.data
    } catch (error) {
      console.error('Failed to list SharePoint files:', error)
      throw error
    }
  },

  async readSharePointFile(path: string, maxSizeMb: number = 50, encoding: string = "utf-8", previewLines: number = 0): Promise<any> {
    try {
      const response = await api.post('/api/mcp/query', {
        query: `Use sharepoint_read_file with path="${path}", max_size_mb=${maxSizeMb}, encoding="${encoding}", preview_lines=${previewLines}`
      }, { timeout: 60000 })
      return response.data
    } catch (error) {
      console.error('Failed to read SharePoint file:', error)
      throw error
    }
  },

  async searchSharePointFiles(query: string, path: string = "", searchType: string = "filename", fileType?: string, maxResults: number = 50, includeContent: boolean = false): Promise<any> {
    try {
      const response = await api.post('/api/mcp/query', {
        query: `Use sharepoint_search_files with query="${query}", path="${path}", search_type="${searchType}"${fileType ? `, file_type="${fileType}"` : ""}, max_results=${maxResults}, include_content=${includeContent}`
      }, { timeout: 60000 })
      return response.data
    } catch (error) {
      console.error('Failed to search SharePoint files:', error)
      throw error
    }
  },

  async extractPdfText(path: string, pages: string = "all", cleanText: boolean = true, preserveStructure: boolean = true, extractTables: boolean = false, maxPages: number = 50): Promise<any> {
    try {
      const response = await api.post('/api/mcp/query', {
        query: `Use extract_pdf_text with path="${path}", pages="${pages}", clean_text=${cleanText}, preserve_structure=${preserveStructure}, extract_tables=${extractTables}, max_pages=${maxPages}`
      }, { timeout: 60000 })
      return response.data
    } catch (error) {
      console.error('Failed to extract PDF text:', error)
      throw error
    }
  },

  // Token management
  async getTokenUsage(sessionId: string = 'default'): Promise<{ session_used: number; session_limit: number; daily_used: number; daily_limit: number; request_limit: number }> {
    try {
      const response = await api.get(`/api/tokens/${sessionId}`)
      return response.data.usage
    } catch (error) {
      console.error('Failed to get token usage:', error)
      return {
        session_used: 0,
        session_limit: 200000,
        daily_used: 0,
        daily_limit: 300000,
        request_limit: 50000
      }
    }
  },

  async getTokenLimits(): Promise<{ session_limit: number; daily_limit: number; request_limit: number }> {
    try {
      const response = await api.get('/api/tokens')
      return response.data.limits
    } catch (error) {
      console.error('Failed to get token limits:', error)
      return {
        session_limit: 200000,
        daily_limit: 300000,
        request_limit: 50000
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
            
            // Don't handle heartbeat here - let the GlobalWebSocketManager handle it
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