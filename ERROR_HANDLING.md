# 🛡️ BroadAxis Error Handling System

## Overview
Comprehensive error handling has been implemented across the entire BroadAxis RFP/RFQ platform to ensure reliability, debugging capability, and user experience.

## 🏗️ Architecture

### 1. Centralized Error Handler (`backend/error_handler.py`)
- **Custom Exception Classes**: Specific error types for different scenarios
- **Logging System**: Structured logging with file and console output
- **Response Formatting**: Consistent error response format for APIs
- **Context Tracking**: Error context for debugging

### 2. Error Classifications
```python
class ErrorType(Enum):
    VALIDATION = "validation"           # Input validation errors
    AUTHENTICATION = "authentication"   # Auth failures
    AUTHORIZATION = "authorization"     # Permission errors
    NOT_FOUND = "not_found"            # Resource not found
    EXTERNAL_API = "external_api"      # Third-party service errors
    DATABASE = "database"              # Data persistence errors
    FILE_OPERATION = "file_operation"  # File system errors
    NETWORK = "network"                # Network connectivity
    SYSTEM = "system"                  # System-level errors
    UNKNOWN = "unknown"                # Unclassified errors
```

## 🔧 Implementation Details

### Backend API (`backend/api.py`)

#### Global Exception Handlers
- **BroadAxisError**: Custom application errors
- **PydanticValidationError**: Request validation failures
- **General Exception**: Catch-all for unexpected errors

#### Enhanced Features
- **Token Management**: Error handling for token limits and counting
- **File Operations**: Validation, size limits, security checks
- **WebSocket**: Connection timeouts, message validation
- **MCP Interface**: Connection status tracking, tool availability

### MCP Server (`ba-server/server.py`)

#### Tool Error Handling
- **@handle_tool_errors**: Decorator for consistent error handling
- **Service Validation**: Check external service availability
- **Input Validation**: Query and parameter validation
- **Graceful Degradation**: Fallback responses when services fail

#### External Service Resilience
- **Pinecone**: Connection validation and error recovery
- **Tavily**: Search query validation and result formatting
- **HuggingFace**: Embedder initialization and fallback
- **arXiv**: Paper search error handling

### Installation & Startup Scripts

#### Enhanced Installation (`install_dependencies.py`)
- **Prerequisite Checking**: Verify required tools are available
- **Timeout Handling**: Prevent hanging installations
- **Graceful Failures**: Continue with non-critical failures
- **Clear Error Messages**: User-friendly error reporting

#### Backend Startup (`backend/run_backend.py`)
- **Dependency Validation**: Check required modules
- **Environment Checking**: Validate configuration
- **Process Management**: Handle server startup failures

## 📊 Error Response Format

### API Responses
```json
{
  "error": true,
  "message": "Human-readable error message",
  "type": "error_classification",
  "status_code": 400,
  "timestamp": "2024-01-01T12:00:00Z",
  "details": {
    "additional_context": "value"
  }
}
```

### WebSocket Responses
```json
{
  "type": "error",
  "message": "Error description",
  "error_type": "validation",
  "status": "error"
}
```

## 🔍 Logging System

### Log Levels
- **INFO**: Normal operations, startup messages
- **WARNING**: Non-critical issues, missing configurations
- **ERROR**: Application errors, failed operations
- **DEBUG**: Detailed debugging information

### Log Files
- **Backend**: `broadaxis_errors.log`
- **MCP Server**: `mcp_server.log`

### Log Format
```
2024-01-01 12:00:00 - BroadAxis - ERROR - Error message with context
```

## 🚨 Error Scenarios Covered

### Input Validation
- Empty or invalid queries
- File type restrictions
- File size limits
- Parameter validation

### External Services
- API key validation
- Service availability checks
- Rate limit handling
- Network timeout management

### File Operations
- Path traversal prevention
- Permission checks
- Disk space validation
- Encoding error handling

### System Resources
- Memory usage monitoring
- Token limit enforcement
- Connection pool management
- Process timeout handling

## 🛠️ Usage Examples

### Custom Error Handling
```python
from error_handler import ValidationError, error_handler

def process_file(filename):
    if not filename:
        raise ValidationError("Filename cannot be empty")
    
    try:
        # File processing logic
        pass
    except Exception as e:
        error_handler.log_error(e, {'filename': filename})
        raise FileOperationError("Failed to process file")
```

### API Error Response
```python
@app.exception_handler(BroadAxisError)
async def handle_custom_error(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_handler.format_error_response(exc)
    )
```

## 📈 Benefits

### For Developers
- **Consistent Error Handling**: Standardized approach across codebase
- **Debugging Support**: Detailed logging and context tracking
- **Maintainability**: Centralized error management

### For Users
- **Clear Error Messages**: User-friendly error descriptions
- **Graceful Degradation**: System continues operating when possible
- **Status Visibility**: Real-time connection and service status

### For Operations
- **Monitoring**: Structured logs for system monitoring
- **Alerting**: Error classification for automated alerts
- **Troubleshooting**: Context-rich error information

## 🔄 Error Recovery Strategies

### Automatic Recovery
- **Retry Logic**: Automatic retries for transient failures
- **Fallback Services**: Alternative service endpoints
- **Circuit Breakers**: Prevent cascading failures

### Manual Recovery
- **Health Checks**: System status endpoints
- **Service Restart**: Graceful service recovery
- **Configuration Reload**: Dynamic configuration updates

## 📋 Monitoring & Alerting

### Health Endpoints
- `/health` - Overall system health
- `/api/status` - Service connection status
- `/api/tokens` - Token usage monitoring

### Error Metrics
- Error rate by endpoint
- Error classification distribution
- Service availability metrics
- Response time monitoring

This comprehensive error handling system ensures the BroadAxis platform is robust, maintainable, and provides excellent user experience even when things go wrong.