import React, { useState, useEffect } from 'react'
import { ProgressTracker as ProgressTrackerType } from '../utils/websocket'

interface ProgressTrackerProps {
  progress: ProgressTrackerType
  onComplete?: () => void
}

const ProgressTracker: React.FC<ProgressTrackerProps> = ({ progress, onComplete }) => {
  const [isVisible, setIsVisible] = useState(true)
  const [elapsedTime, setElapsedTime] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(Date.now() - progress.startTime)
    }, 100)

    return () => clearInterval(timer)
  }, [progress.startTime])

  useEffect(() => {
    if (progress.progress >= 100) {
      setTimeout(() => {
        setIsVisible(false)
        onComplete?.()
      }, 1000)
    }
  }, [progress.progress, onComplete])

  const formatTime = (ms: number): string => {
    const seconds = Math.floor(ms / 1000)
    if (seconds < 60) return `${seconds}s`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}m ${remainingSeconds}s`
  }



  const getProgressIcon = (type: ProgressTrackerType['type']): string => {
    switch (type) {
      case 'upload': return '📤'
      case 'processing': return '⚙️'
      case 'search': return '🔍'
      case 'generation': return '✨'
      case 'tool_execution': return '🔧'
      default: return '🔄'
    }
  }

  const getProgressPhase = (progress: number, message: string): string => {
    if (progress <= 30) return 'Processing Request'
    if (progress <= 60) return 'Analyzing Content'
    if (progress <= 90) return 'Generating Response'
    return 'Finalizing'
  }

  const getPhaseProgress = (progress: number): number => {
    if (progress <= 30) return Math.round((progress / 30) * 25)
    if (progress <= 60) return Math.round(25 + ((progress - 30) / 30) * 25)
    if (progress <= 90) return Math.round(50 + ((progress - 60) / 30) * 25)
    return Math.round(75 + ((progress - 90) / 10) * 25)
  }

  const getProgressColor = (progress: number): string => {
    if (progress <= 30) return 'bg-blue-500'
    if (progress <= 60) return 'bg-purple-500'
    if (progress <= 90) return 'bg-yellow-500'
    return 'bg-green-500'
  }

  if (!isVisible) return null

  return (
    <div className="bg-white/95 backdrop-blur-sm border border-blue-200/50 rounded-lg shadow-sm p-3 mb-2 transition-all duration-300 ease-in-out">
      <div className="flex items-center space-x-3">
        {/* Icon */}
        <span className="text-lg">{getProgressIcon(progress.type)}</span>
        
        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1">
            <h4 className="font-medium text-sm text-gray-900 dark:text-white truncate">
              {getProgressPhase(progress.progress, progress.message)}
            </h4>
            <div className="text-xs text-gray-500 dark:text-gray-400 ml-2">
              {formatTime(elapsedTime)}
            </div>
          </div>
          
          <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 truncate">
            {progress.message}
          </p>
          
          {/* Compact Progress Bar */}
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5 mb-1">
            <div
              className={`h-1.5 rounded-full transition-all duration-300 ease-out ${getProgressColor(progress.progress)}`}
              style={{ width: `${getPhaseProgress(progress.progress)}%` }}
            />
          </div>
          
          {/* Progress indicator - show phase and percentage */}
          <div className="text-xs text-gray-500 dark:text-gray-400 text-right">
            {getProgressPhase(progress.progress, progress.message)} • {getPhaseProgress(progress.progress)}%
          </div>
        </div>
      </div>
    </div>
  )
}

export default ProgressTracker
