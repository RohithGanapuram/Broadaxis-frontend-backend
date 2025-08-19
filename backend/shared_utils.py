"""
Shared utilities for BroadAxis RFP/RFQ Management Platform
"""

import os
import sys
import re
import logging
from typing import Dict, Set
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shared_utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SharedUtils')

# MCP Resource Templates for SharePoint
SERVER_FOLDER_TEMPLATE = "sharepoint+folder://sites/{site}/drives/{drive}/root:/{folder_path}"
SERVER_FILE_TEMPLATE = "sharepoint://sites/{site}/drives/{drive}/root:/{folder_path}/{filename}"

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".doc"}

def _parse_folder_uri(uri: str) -> dict:
    """Parse SharePoint folder URI into components"""
    try:
        if not uri.startswith("sharepoint+folder://"):
            raise ValueError("Invalid folder URI format")
        
        # Remove prefix and split
        path_part = uri.replace("sharepoint+folder://", "")
        parts = path_part.split("/")
        
        if len(parts) < 4:
            raise ValueError("Invalid folder URI structure")
        
        # Extract components
        site = parts[0]  # sites/{site}
        drive = parts[2]  # drives/{drive}
        folder_path = "/".join(parts[4:])  # root:/{folder_path}
        
        # Clean up site and drive
        site = site.replace("sites/", "")
        drive = drive.replace("drives/", "")
        folder_path = folder_path.replace("root:/", "")
        
        return {
            "site": site,
            "drive": drive,
            "folder_path": folder_path
        }
    except Exception as e:
        logger.error(f"Error parsing folder URI {uri}: {e}")
        raise ValueError(f"Invalid folder URI: {uri}")

def _parse_file_uri(uri: str) -> dict:
    """Parse SharePoint file URI into components"""
    try:
        if not uri.startswith("sharepoint://"):
            raise ValueError("Invalid file URI format")
        
        # Remove prefix and split
        path_part = uri.replace("sharepoint://", "")
        parts = path_part.split("/")
        
        if len(parts) < 5:
            raise ValueError("Invalid file URI structure")
        
        # Extract components
        site = parts[0]  # sites/{site}
        drive = parts[2]  # drives/{drive}
        folder_path = "/".join(parts[4:-1])  # root:/{folder_path}
        filename = parts[-1]  # {filename}
        
        # Clean up site and drive
        site = site.replace("sites/", "")
        drive = drive.replace("drives/", "")
        folder_path = folder_path.replace("root:/", "")
        
        return {
            "site": site,
            "drive": drive,
            "folder_path": folder_path,
            "filename": filename
        }
    except Exception as e:
        logger.error(f"Error parsing file URI {uri}: {e}")
        raise ValueError(f"Invalid file URI: {uri}")

def sanitize_filename(name: str) -> str:
    """Remove all non-alphanumeric, dash, underscore characters"""
    return re.sub(r'[^a-zA-Z0-9-_]', '_', name)

def validate_file_extension(filename: str) -> bool:
    """Validate if file extension is allowed"""
    file_ext = os.path.splitext(filename.lower())[1]
    return file_ext in ALLOWED_EXTENSIONS

def get_sharepoint_manager():
    """Get SharePointManager instance with proper import handling"""
    try:
        from .api import SharePointManager
        return SharePointManager()
    except ImportError:
        # For server.py usage
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
        from api import SharePointManager
        return SharePointManager()

def generate_mcp_uris(site: str, drive: str, folder_path: str, filename: str) -> Dict[str, str]:
    """Generate MCP URIs for SharePoint resources"""
    folder_uri = SERVER_FOLDER_TEMPLATE.format(
        site=site,
        drive=drive,
        folder_path=folder_path
    )
    file_uri = SERVER_FILE_TEMPLATE.format(
        site=site,
        drive=drive,
        folder_path=folder_path,
        filename=filename
    )
    return {
        "folder_uri": folder_uri,
        "file_uri": file_uri
    }
