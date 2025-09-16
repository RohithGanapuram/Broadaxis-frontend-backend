import React from 'react'

interface TokenDisplayProps {
  tokenUsage: {
    total_tokens: number
    input_tokens: number
    output_tokens: number
    model_used: string
    request_id: string
  }
  compact?: boolean
}

const TokenDisplay: React.FC<TokenDisplayProps> = ({ tokenUsage, compact = false }) => {
  if (!tokenUsage || tokenUsage.total_tokens === 0) return null

  const formatModelName = (model: string) => {
    if (model.includes('haiku')) return 'Haiku'
    if (model.includes('sonnet')) return 'Sonnet'
    if (model.includes('opus')) return 'Opus'
    return model.split('-').pop() || model
  }

  if (compact) {
    return (
      <div className="flex items-center space-x-2 text-xs text-gray-500 dark:text-gray-400 mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
        <span className="flex items-center space-x-1">
          <span className="text-blue-500">🔢</span>
          <span>{tokenUsage.total_tokens.toLocaleString()} tokens</span>
        </span>
        <span className="text-gray-300">•</span>
        <span className="text-green-500">{formatModelName(tokenUsage.model_used)}</span>
        <span className="text-gray-300">•</span>
        <span className="text-purple-500">ID: {tokenUsage.request_id.slice(0, 8)}</span>
      </div>
    )
  }

  return (
    <div className="mt-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center space-x-2">
          <span className="text-blue-500">🔢</span>
          <span>Token Usage</span>
        </h4>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {formatModelName(tokenUsage.model_used)}
        </span>
      </div>
      
      <div className="grid grid-cols-3 gap-3 text-xs">
        <div className="text-center">
          <div className="text-lg font-semibold text-blue-600 dark:text-blue-400">
            {tokenUsage.total_tokens.toLocaleString()}
          </div>
          <div className="text-gray-500 dark:text-gray-400">Total</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-green-600 dark:text-green-400">
            {tokenUsage.input_tokens.toLocaleString()}
          </div>
          <div className="text-gray-500 dark:text-gray-400">Input</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-purple-600 dark:text-purple-400">
            {tokenUsage.output_tokens.toLocaleString()}
          </div>
          <div className="text-gray-500 dark:text-gray-400">Output</div>
        </div>
      </div>
      
      <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 text-center">
        Request ID: {tokenUsage.request_id}
      </div>
    </div>
  )
}

export default TokenDisplay

