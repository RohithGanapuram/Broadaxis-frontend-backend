import { ChatWebSocket } from './api'

// Global WebSocket connection that persists across page navigation
class GlobalWebSocketManager {
  private ws: ChatWebSocket | null = null
  private isConnected: boolean = false
  private messageHandlers: Set<(message: any) => void> = new Set()
  private errorHandlers: Set<(error: Event) => void> = new Set()
  private closeHandlers: Set<() => void> = new Set()

  async connect(): Promise<void> {
    if (this.ws?.isConnected()) {
      this.isConnected = true
      return
    }

    this.ws = new ChatWebSocket(
      (message) => {
        this.messageHandlers.forEach(handler => handler(message))
      },
      (error) => {
        this.isConnected = false
        this.errorHandlers.forEach(handler => handler(error))
      },
      () => {
        this.isConnected = false
        this.closeHandlers.forEach(handler => handler())
      }
    )

    await this.ws.connect()
    this.isConnected = true
  }

  addMessageHandler(handler: (message: any) => void): void {
    this.messageHandlers.add(handler)
  }

  removeMessageHandler(handler: (message: any) => void): void {
    this.messageHandlers.delete(handler)
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
      this.ws.sendMessage(message)
    }
  }

  getConnectionStatus(): boolean {
    return this.isConnected && this.ws?.isConnected() || false
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.disconnect()
      this.ws = null
      this.isConnected = false
    }
  }
}

// Export singleton instance
export const globalWebSocket = new GlobalWebSocketManager()