export interface ChatMessage {
  id: string
  type: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  isLoading?: boolean
}

export interface FileInfo {
  status: string
  filename: string
  size: number
  message: string
  analysis?: string
}

export interface ChatRequest {
  query: string
  enabled_tools: string[]
  model: string
}

export interface ChatResponse {
  response: string
  status: string
}

export interface Tool {
  name: string
  description: string
  input_schema: any
}

export interface Prompt {
  name: string
  description: string
  arguments: any[]
}

export interface UploadResponse {
  status: string
  filename: string
  size: number
  message: string
  analysis: string
}

export interface WebSocketMessage {
  type: 'status' | 'response' | 'error'
  message: string
  status?: string
}

export interface AppSettings {
  model: string
  enabledTools: string[]
  autoAnalyze: boolean
  theme: 'light' | 'dark'
}

export interface GeneratedFile {
  filename: string
  file_path: string
  file_size: number
  modified_at: string
  mime_type: string
  type: string
}
