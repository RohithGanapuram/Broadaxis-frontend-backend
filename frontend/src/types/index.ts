export interface ChatMessage {
  id: string
  type: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  isLoading?: boolean
  tokens_used?: number
  tokens_remaining?: number
  usage?: {
    session_used: number
    session_limit: number
    daily_used: number
    daily_limit: number
  }
}

export interface FileInfo {
  status: string
  filename: string
  size: number
  message: string
  analysis?: string
}



export interface ChatResponse {
  response: string
  status: string
  tokens_used?: number
  tokens_remaining?: number
  usage?: {
    session_used: number
    session_limit: number
    daily_used: number
    daily_limit: number
  }
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
  type: 'status' | 'response' | 'error' | 'progress' | 'connection' | 'heartbeat' | 'timeout' | 'pong'
  message: string
  status?: string
  timestamp?: string
  session_id?: string
  reconnect_advice?: string
  progress?: number
  tokens_used?: number
  tokens_remaining?: number
  usage?: {
    session_used: number
    session_limit: number
    daily_used: number
    daily_limit: number
  }
}

export interface AppSettings {
  model: string
  enabledTools: string[]
  autoAnalyze: boolean
  theme: 'light' | 'dark'
}

export interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: Date
  updatedAt: Date
}

export interface GeneratedFile {
  filename: string
  file_size: number
  modified_at: string
  type: string
}

export interface EmailAttachment {
  filename: string
  file_path?: string
  file_size?: number
  download_date: string
  type: 'file' | 'link'
  url?: string
  domain?: string
  email_subject?: string
  email_sender?: string
  email_date?: string
  sharepoint_path?: string
}

export interface FetchedEmail {
  email_id: string
  sender: string
  subject: string
  date: string
  account: string
  attachments: EmailAttachment[]
  has_rfp_keywords: boolean
}

export interface EmailFetchRequest {
  email_accounts?: string[]
  use_real_email?: boolean
  use_graph_api?: boolean
}

export interface EmailFetchResponse {
  status: string
  message: string
  emails_found: number
  attachments_downloaded: number
  fetched_emails: FetchedEmail[]
}
