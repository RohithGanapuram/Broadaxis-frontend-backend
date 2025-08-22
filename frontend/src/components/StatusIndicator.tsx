import React, { useState, useEffect } from 'react'
import { WebSocketStatusUpdate } from '../utils/websocket'

interface StatusIndicatorProps {
  status: WebSocketStatusUpdate
  onDismiss?: () => void
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status, onDismiss }) => {
  const [isVisible, setIsVisible] = useState(true)
  const [isAnimating, setIsAnimating] = useState(false)

  useEffect(() => {
    setIsAnimating(true)
    const timer = setTimeout(() => setIsAnimating(false), 500)

    // Auto-dismiss after 5 seconds for info/success messages
    if (status.status === 'info' || status.status === 'success') {
      const dismissTimer = setTimeout(() => {
        setIsVisible(false)
        onDismiss?.()
      }, 5000)
      return () => clearTimeout(dismissTimer)
    }

    return () => clearTimeout(timer)
  }, [status, onDismiss])

  const getStatusIcon = (statusType: string): string => {
    switch (statusType) {
      case 'success': return '✅'
      case 'error': return '❌'
      case 'warning': return '⚠️'
      case 'info': return 'ℹ️'
      default: return '📢'
    }
  }

  const getStatusColor = (statusType: string): string => {
    switch (statusType) {
      case 'success': return 'bg-green-50 border-green-200 text-green-800 dark:bg-green-900/20 dark:border-green-800 dark:text-green-400'
      case 'error': return 'bg-red-50 border-red-200 text-red-800 dark:bg-red-900/20 dark:border-red-800 dark:text-red-400'
      case 'warning': return 'bg-yellow-50 border-yellow-200 text-yellow-800 dark:bg-yellow-900/20 dark:border-yellow-800 dark:text-yellow-400'
      case 'info': return 'bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-400'
      default: return 'bg-gray-50 border-gray-200 text-gray-800 dark:bg-gray-900/20 dark:border-gray-800 dark:text-gray-400'
    }
  }

  const getStatusAnimation = (statusType: string): string => {
    switch (statusType) {
      case 'success': return 'animate-pulse'
      case 'error': return 'animate-bounce'
      case 'warning': return 'animate-pulse'
      case 'info': return 'animate-pulse'
      default: return ''
    }
  }

  if (!isVisible) return null

  return (
    <div
      className={`border rounded-lg p-4 mb-3 transition-all duration-300 ease-in-out ${getStatusColor(status.status)} ${
        isAnimating ? getStatusAnimation(status.status) : ''
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3">
          <span className="text-xl mt-0.5">{getStatusIcon(status.status)}</span>
          <div className="flex-1">
            <p className="font-medium">{status.message}</p>
            {status.timestamp && (
              <p className="text-xs opacity-75 mt-1">
                {new Date(status.timestamp).toLocaleTimeString()}
              </p>
            )}
          </div>
        </div>
        {(status.status === 'error' || status.status === 'warning') && (
          <button
            onClick={() => {
              setIsVisible(false)
              onDismiss?.()
            }}
            className="text-current opacity-60 hover:opacity-100 transition-opacity"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  )
}

export default StatusIndicator
