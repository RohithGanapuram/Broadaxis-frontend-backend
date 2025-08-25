import { ChatWebSocket } from './api'

// Enhanced WebSocket message types
export interface WebSocketProgressUpdate {
  type: 'progress'
  message: string
  progress: number
  step?: string
  total_steps?: number
  current_step?: number
}

export interface WebSocketStatusUpdate {
  type: 'status'
  message: string
  status: 'info' | 'success' | 'warning' | 'error'
  timestamp?: string
}

export interface WebSocketResponseUpdate {
  type: 'response'
  message: string
  tokens_used?: number
  tokens_remaining?: number
  usage?: {
    session_used: number
    session_limit: number
    daily_used: number
    daily_limit: number
  }
}

export interface WebSocketErrorUpdate {
  type: 'error'
  message: string
  error_type?: string
  retry_after?: number
}

// Progress tracking interface
export interface ProgressTracker {
  id: string
  type: 'upload' | 'processing' | 'search' | 'generation' | 'tool_execution'
  title: string
  progress: number
  message: string
  startTime: number
  estimatedTime?: number
  steps?: string[]
  currentStep?: number
}

// Global WebSocket connection that persists across page navigation
class GlobalWebSocketManager {
  private ws: ChatWebSocket | null = null
  private isConnected: boolean = false
  private messageHandlers: Set<(message: any) => void> = new Set()
  private errorHandlers: Set<(error: Event) => void> = new Set()
  private closeHandlers: Set<() => void> = new Set()
  private progressHandlers: Set<(progress: ProgressTracker) => void> = new Set()
  private statusHandlers: Set<(status: WebSocketStatusUpdate) => void> = new Set()
  private reconnectAttempts: number = 0
  private maxReconnectAttempts: number = 5
  private reconnectDelay: number = 1000
  private activeProgress: Map<string, ProgressTracker> = new Map()
  private heartbeatInterval: NodeJS.Timeout | null = null

  async connect(): Promise<void> {
    if (this.ws?.isConnected()) {
      this.isConnected = true
      return
    }

    try {
      this.ws = new ChatWebSocket(
        (message) => {
          this.handleMessage(message)
        },
        (error) => {
          this.isConnected = false
          this.errorHandlers.forEach(handler => handler(error))
          this.attemptReconnect()
        },
        () => {
          this.isConnected = false
          this.closeHandlers.forEach(handler => handler())
          this.attemptReconnect()
        }
      )

      await this.ws.connect()
      this.isConnected = true
      this.reconnectAttempts = 0
      this.startHeartbeat()
      
      // Send initial connection message
      this.sendMessage({
        type: 'connection',
        message: 'Client connected',
        timestamp: new Date().toISOString()
      })
    } catch (error) {
      console.error('WebSocket connection failed:', error)
      this.attemptReconnect()
    }
  }

  private handleMessage(message: any): void {
    // Handle different message types with specific logic
    switch (message.type) {
      case 'progress':
        this.handleProgressUpdate(message)
        break
      case 'status':
        this.handleStatusUpdate(message)
        break
      case 'response':
        this.handleResponseUpdate(message)
        break
      case 'error':
        this.handleErrorUpdate(message)
        break
      case 'heartbeat':
        this.handleHeartbeat(message)
        break
      case 'connection':
        this.handleConnectionUpdate(message)
        break
      default:
        // Pass through to general message handlers
        this.messageHandlers.forEach(handler => handler(message))
    }
  }

  private handleProgressUpdate(progress: WebSocketProgressUpdate): void {
    // Use a single progress tracker for the entire request lifecycle
    const progressId = 'request-progress'
    
    // Get existing tracker or create new one
    const existingTracker = this.activeProgress.get(progressId)
    const startTime = existingTracker?.startTime || Date.now()
    
    const tracker: ProgressTracker = {
      id: progressId,
      type: this.determineProgressType(progress.message),
      title: this.extractProgressTitle(progress.message),
      progress: progress.progress,
      message: progress.message,
      startTime: startTime,
      currentStep: progress.current_step,
      total_steps: progress.total_steps
    }

    this.activeProgress.set(progressId, tracker)
    this.progressHandlers.forEach(handler => handler(tracker))

    // Also pass to general message handlers
    this.messageHandlers.forEach(handler => handler(progress))
  }

  private handleStatusUpdate(status: WebSocketStatusUpdate): void {
    this.statusHandlers.forEach(handler => handler(status))
    this.messageHandlers.forEach(handler => handler(status))
  }

  private handleResponseUpdate(response: WebSocketResponseUpdate): void {
    // Clear any active progress for this response
    this.activeProgress.delete('request-progress')
    this.messageHandlers.forEach(handler => handler(response))
  }

  private handleErrorUpdate(error: WebSocketErrorUpdate): void {
    // Clear progress on error
    this.activeProgress.delete('request-progress')
    this.messageHandlers.forEach(handler => handler(error))
  }

  private handleHeartbeat(heartbeat: any): void {
    // Don't send pong response here - backend will handle it
    // Just log and pass to handlers
    console.log('Heartbeat received:', heartbeat)
    this.messageHandlers.forEach(handler => handler(heartbeat))
  }

  private handleConnectionUpdate(connection: any): void {
    console.log('Connection update:', connection.message)
    this.messageHandlers.forEach(handler => handler(connection))
  }

  private determineProgressType(message: string): ProgressTracker['type'] {
    const lowerMessage = message.toLowerCase()
    if (lowerMessage.includes('upload') || lowerMessage.includes('file')) return 'upload'
    if (lowerMessage.includes('search') || lowerMessage.includes('query')) return 'search'
    if (lowerMessage.includes('generate') || lowerMessage.includes('create')) return 'generation'
    if (lowerMessage.includes('tool') || lowerMessage.includes('execute')) return 'tool_execution'
    return 'processing'
  }

  private extractProgressTitle(message: string): string {
    // Extract meaningful title from progress message
    if (message.includes('Rate limit')) return 'Rate Limiting'
    if (message.includes('Anthropic servers')) return 'Server Processing'
    if (message.includes('Executing')) return 'Tool Execution'
    if (message.includes('Searching')) return 'Searching'
    if (message.includes('Generating')) return 'Generating Response'
    return 'Processing'
  }

  private async attemptReconnect(): Promise<void> {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
    
    setTimeout(async () => {
      try {
        await this.connect()
      } catch (error) {
        console.error('Reconnection failed:', error)
      }
    }, delay)
  }

  private startHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
    }

    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected && this.ws?.isConnected()) {
        // Add a small random delay to prevent exact timing conflicts
        const delay = Math.random() * 1000 // 0-1 second random delay
        setTimeout(() => {
          if (this.isConnected && this.ws?.isConnected()) {
            this.sendMessage({ type: 'ping', timestamp: Date.now() })
          }
        }, delay)
      }
    }, 30000) // Send heartbeat every 30 seconds
  }

  // Enhanced handler registration
  addMessageHandler(handler: (message: any) => void): void {
    this.messageHandlers.add(handler)
  }

  removeMessageHandler(handler: (message: any) => void): void {
    this.messageHandlers.delete(handler)
  }

  addProgressHandler(handler: (progress: ProgressTracker) => void): void {
    this.progressHandlers.add(handler)
  }

  removeProgressHandler(handler: (progress: ProgressTracker) => void): void {
    this.progressHandlers.delete(handler)
  }

  addStatusHandler(handler: (status: WebSocketStatusUpdate) => void): void {
    this.statusHandlers.add(handler)
  }

  removeStatusHandler(handler: (status: WebSocketStatusUpdate) => void): void {
    this.statusHandlers.delete(handler)
  }

  addErrorHandler(handler: (error: Event) => void): void {
    this.errorHandlers.add(handler)
  }

  removeErrorHandler(handler: (error: Event) => void): void {
    this.errorHandlers.delete(handler)
  }

  addCloseHandler(handler: () => void): void {
    this.closeHandlers.add(handler)
  }

  removeCloseHandler(handler: () => void): void {
    this.closeHandlers.delete(handler)
  }

  sendMessage(message: any): void {
    if (this.ws?.isConnected()) {
      // Validate message before sending
      if (message && typeof message === 'object') {
        this.ws.sendMessage(message)
      } else {
        console.warn('Invalid message format, not sending:', message)
      }
    } else {
      console.warn('WebSocket not connected, message not sent:', message)
    }
  }

  getConnectionStatus(): boolean {
    return this.isConnected && this.ws?.isConnected() || false
  }

  getActiveProgress(): ProgressTracker[] {
    return Array.from(this.activeProgress.values())
  }

  clearProgress(progressId?: string): void {
    if (progressId) {
      this.activeProgress.delete(progressId)
    } else {
      this.activeProgress.clear()
    }
  }

  disconnect(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }

    if (this.ws) {
      this.ws.disconnect()
      this.ws = null
      this.isConnected = false
    }

    this.activeProgress.clear()
  }
}

// Export singleton instance
export const globalWebSocket = new GlobalWebSocketManager()