"""
Centralized Error Handling for BroadAxis RFP/RFQ Platform
"""
import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ErrorType(Enum):
    """Error type classifications"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    FILE_OPERATION = "file_operation"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"

class BroadAxisError(Exception):
    """Base exception class for BroadAxis platform"""
    def __init__(self, message: str, error_type: ErrorType = ErrorType.UNKNOWN, 
                 details: Optional[Dict] = None, status_code: int = 500):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)

class ValidationError(BroadAxisError):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.VALIDATION, details, 400)

class AuthenticationError(BroadAxisError):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.AUTHENTICATION, details, 401)

class NotFoundError(BroadAxisError):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.NOT_FOUND, details, 404)

class ExternalAPIError(BroadAxisError):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.EXTERNAL_API, details, 502)

class FileOperationError(BroadAxisError):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.FILE_OPERATION, details, 500)

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for error handling"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('broadaxis_errors.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BroadAxis')
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log error with context information"""
        error_info = {
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        if isinstance(error, BroadAxisError):
            error_info.update({
                'error_classification': error.error_type.value,
                'details': error.details,
                'status_code': error.status_code,
                'timestamp': error.timestamp
            })
        
        self.logger.error(f"Error occurred: {error_info}")
        return error_info
    
    def format_error_response(self, error: Exception, include_traceback: bool = False) -> Dict[str, Any]:
        """Format error for API response"""
        if isinstance(error, BroadAxisError):
            response = {
                'error': True,
                'message': error.message,
                'type': error.error_type.value,
                'status_code': error.status_code,
                'timestamp': error.timestamp
            }
            if error.details:
                response['details'] = error.details
        else:
            response = {
                'error': True,
                'message': 'An unexpected error occurred',
                'type': ErrorType.UNKNOWN.value,
                'status_code': 500,
                'timestamp': datetime.now().isoformat()
            }
        
        if include_traceback:
            response['traceback'] = traceback.format_exc()
        
        return response
    
    def handle_external_api_error(self, service_name: str, error: Exception) -> ExternalAPIError:
        """Handle external API errors with context"""
        details = {
            'service': service_name,
            'original_error': str(error),
            'error_type': type(error).__name__
        }
        return ExternalAPIError(f"External API error from {service_name}", details)
    
    def handle_file_operation_error(self, operation: str, file_path: str, error: Exception) -> FileOperationError:
        """Handle file operation errors with context"""
        details = {
            'operation': operation,
            'file_path': file_path,
            'original_error': str(error),
            'error_type': type(error).__name__
        }
        return FileOperationError(f"File operation '{operation}' failed", details)

# Global error handler instance
error_handler = ErrorHandler()