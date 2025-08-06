import axios from 'axios'
import { ChatRequest, ChatResponse, FileInfo, Tool, Prompt, UploadResponse, GeneratedFile } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

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
  async uploadFile(file: File): Promise<FileInfo> {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000 // 60 seconds for file upload
    })
    return response.data
  },



  // Tools and prompts
  async getAvailableTools(): Promise<{ tools: Tool[]; status: string }> {
    try {
      const response = await api.get('/api/tools', { timeout: 120000 })
      return response.data
    } catch (error) {
      console.error('Failed to get tools:', error)
      // Return fallback tools
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
        status: "fallback"
      }
    }
  },

  async getAvailablePrompts(): Promise<{ prompts: Prompt[]; status: string }> {
    try {
      const response = await api.get('/api/prompts', { timeout: 120000 })
      return response.data
    } catch (error) {
      console.error('Failed to get prompts:', error)
      // Return fallback prompts
      return {
        prompts: [
          { name: "Step-2: Executive Summary", description: "Generate executive summary of RFP documents", arguments: [] },
          { name: "Step-3: Go/No-Go Recommendation", description: "Provide Go/No-Go analysis", arguments: [] },
          { name: "Step-4: Generate Proposal", description: "Generate capability statement", arguments: [] }
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
