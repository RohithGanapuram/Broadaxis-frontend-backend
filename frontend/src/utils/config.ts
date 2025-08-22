// Frontend configuration and feature documentation
export const APP_CONFIG = {
  // Rate Limiting Information
  rateLimiting: {
    description: "The system now includes advanced rate limiting to ensure stable performance:",
    features: [
      "Server-wide RPM (Requests Per Minute) limiting",
      "In-flight request concurrency control", 
      "Per-session throttling to prevent request bunching",
      "Automatic retry with exponential backoff",
      "Respect for API Retry-After headers",
      "Tool execution concurrency limits"
    ],
    userGuidance: [
      "Complex queries may take longer due to rate limiting",
      "Multiple concurrent users may experience slight delays",
      "The system will automatically retry failed requests",
      "Progress indicators show when rate limiting is active"
    ]
  },
  
  // Timeout Configuration
  timeouts: {
    defaultRequest: 30000, // 30 seconds
    fileUpload: 60000,     // 60 seconds
    websocketMessage: 300000, // 5 minutes
  },
  
  // Feature Flags
  features: {
    realTimeProgress: true,
    rateLimitIndicators: true,
    enhancedErrorHandling: true,
    tokenUsageTracking: true,
    webSocketReconnection: true
  }
}

// Environment-specific settings
export const getApiConfig = () => {
  const isDevelopment = import.meta.env?.MODE === 'development'
  
  return {
    baseUrl: import.meta.env?.VITE_API_URL || 'http://localhost:8000',
    wsUrl: import.meta.env?.VITE_WS_URL || 'ws://localhost:8000',
    debugMode: isDevelopment,
    logLevel: isDevelopment ? 'debug' : 'error'
  }
}
