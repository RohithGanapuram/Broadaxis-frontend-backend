"""
SharePoint API for BroadAxis RFP/RFQ Management Platform
"""

import os
import time
import re
import urllib.parse
import requests
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi import HTTPException

from error_handler import error_handler

# Create router for SharePoint endpoints
sharepoint_router = APIRouter(prefix="/api", tags=["sharepoint"])

class SharePointManager:
    def __init__(self):
        self.graph_config = {
            'client_id': os.getenv('GRAPH_CLIENT_ID'),
            'client_secret': os.getenv('GRAPH_CLIENT_SECRET'),
            'tenant_id': os.getenv('GRAPH_TENANT_ID'),
            'site_url': os.getenv('SHAREPOINT_SITE_URL', 'broadaxis.sharepoint.com:/sites/RFI-project'),
            'folder_path': os.getenv('SHAREPOINT_FOLDER_PATH', 'Documents')
        }
        # Cache for tokens and site/drive info
        self._token_cache = {'token': None, 'expires_at': 0}
        self._site_cache = {'site_id': None, 'drive_id': None, 'expires_at': 0}
        self._session = requests.Session()
        self._session.timeout = 10

    def get_graph_access_token(self):
        """Get access token for Microsoft Graph API with caching"""
        now = time.time()
        
        # Return cached token if still valid (with 5 min buffer)
        if self._token_cache['token'] and now < self._token_cache['expires_at'] - 300:
            return self._token_cache['token']
        
        try:
            token_url = f"https://login.microsoftonline.com/{self.graph_config['tenant_id']}/oauth2/v2.0/token"
            data = {
                'client_id': self.graph_config['client_id'],
                'client_secret': self.graph_config['client_secret'],
                'scope': 'https://graph.microsoft.com/.default',
                'grant_type': 'client_credentials'
            }

            response = self._session.post(token_url, data=data)
            response.raise_for_status()
            token_data = response.json()
            
            # Cache token with expiration
            self._token_cache['token'] = token_data.get('access_token')
            self._token_cache['expires_at'] = now + token_data.get('expires_in', 3600)
            
            return self._token_cache['token']
        except Exception as e:
            error_handler.log_error(e, {'operation': 'get_graph_access_token'})
            return None

    def _get_site_and_drive_info(self):
        """Get and cache site and drive information"""
        now = time.time()
        
        # Return cached info if still valid (cache for 1 hour)
        if (self._site_cache['site_id'] and self._site_cache['drive_id'] and 
            now < self._site_cache['expires_at']):
            return self._site_cache['site_id'], self._site_cache['drive_id']
        
        access_token = self.get_graph_access_token()
        if not access_token:
            return None, None
        
        headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
        
        try:
            # Get site info
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.graph_config['site_url']}"
            site_response = self._session.get(site_url, headers=headers)
            site_response.raise_for_status()
            site_id = site_response.json()['id']
            
            # Get drive info
            drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
            drive_response = self._session.get(drive_url, headers=headers)
            drive_response.raise_for_status()
            drive_id = drive_response.json()['id']
            
            # Cache the info
            self._site_cache.update({
                'site_id': site_id,
                'drive_id': drive_id,
                'expires_at': now + 3600  # 1 hour cache
            })
            
            return site_id, drive_id
        except Exception as e:
            error_handler.log_error(e, {'operation': 'get_site_and_drive_info'})
            return None, None

    def get_sharepoint_files(self):
        """Get files and folders from SharePoint"""
        try:
            site_id, drive_id = self._get_site_and_drive_info()
            if not site_id or not drive_id:
                return {"status": "error", "message": "Failed to get site/drive information", "files": []}

            access_token = self.get_graph_access_token()
            headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}

            # Get files from root folder
            files_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"
            files_response = self._session.get(files_url, headers=headers)

            if files_response.status_code != 200:
                return {"status": "error", "message": "Failed to get SharePoint files", "files": []}

            files_data = files_response.json()
            sharepoint_files = []

            # Process items efficiently
            for item in files_data.get('value', []):
                file_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'type': 'folder' if 'folder' in item else 'file',
                    'size': item.get('size', 0),
                    'modified': item.get('lastModifiedDateTime', ''),
                    'download_url': item.get('@microsoft.graph.downloadUrl', ''),
                    'web_url': item.get('webUrl', ''),
                    'path': item.get('parentReference', {}).get('path', '') + '/' + item['name']
                }

                # Only get folder contents if needed (avoid unnecessary API calls)
                if file_info['type'] == 'folder':
                    folder_files = self.get_folder_contents(site_id, drive_id, item['id'], headers)
                    file_info['children'] = folder_files

                sharepoint_files.append(file_info)

            return {
                "status": "success",
                "message": f"Successfully retrieved {len(sharepoint_files)} items from SharePoint",
                "files": sharepoint_files
            }

        except Exception as e:
            error_handler.log_error(e, {'operation': 'get_sharepoint_files'})
            return {"status": "error", "message": str(e), "files": []}

    def get_folder_contents(self, site_id, drive_id, folder_id, headers):
        """Get contents of a specific folder"""
        try:
            folder_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{folder_id}/children"
            folder_response = requests.get(folder_url, headers=headers)

            if folder_response.status_code != 200:
                return []

            folder_data = folder_response.json()
            folder_files = []

            for item in folder_data.get('value', []):
                file_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'type': 'folder' if 'folder' in item else 'file',
                    'size': item.get('size', 0),
                    'modified': item.get('lastModifiedDateTime', ''),
                    'download_url': item.get('@microsoft.graph.downloadUrl', ''),
                    'web_url': item.get('webUrl', ''),
                    'path': item.get('parentReference', {}).get('path', '') + '/' + item['name']
                }
                folder_files.append(file_info)

            return folder_files

        except Exception as e:
            print(f"Error getting folder contents: {e}")
            return []

    def get_folder_contents_by_path(self, folder_path: str):
        """Get contents of a specific folder by path"""
        try:
            site_id, drive_id = self._get_site_and_drive_info()
            if not site_id or not drive_id:
                return {"status": "error", "message": "Failed to get site/drive information", "files": []}

            access_token = self.get_graph_access_token()
            headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}

            # Get files from the specific folder path
            if folder_path and folder_path != "":
                files_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{folder_path}:/children"
            else:
                files_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"

            files_response = self._session.get(files_url, headers=headers)

            if files_response.status_code != 200:
                return {"status": "error", "message": f"Failed to get folder contents: {files_response.status_code}", "files": []}

            files_data = files_response.json()
            folder_items = []

            for item in files_data.get('value', []):
                file_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'type': 'folder' if 'folder' in item else 'file',
                    'size': item.get('size', 0),
                    'modified': item.get('lastModifiedDateTime', ''),
                    'download_url': item.get('@microsoft.graph.downloadUrl', ''),
                    'web_url': item.get('webUrl', ''),
                    'path': f"{folder_path}/{item['name']}" if folder_path else item['name']
                }
                folder_items.append(file_info)

            return {
                "status": "success",
                "message": f"Successfully retrieved {len(folder_items)} items from folder: {folder_path}",
                "files": folder_items
            }

        except Exception as e:
            error_handler.log_error(e, {'operation': 'get_folder_contents_by_path', 'folder_path': folder_path})
            return {"status": "error", "message": str(e), "files": []}

    def upload_file_to_sharepoint(self, file_content: bytes, filename: str, folder_path: str):
        """Upload a file to SharePoint"""
        try:
            access_token = self.get_graph_access_token()
            if not access_token:
                return {"status": "error", "message": "Failed to get access token"}

            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/octet-stream'
            }

            # Get site and drive info
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.graph_config['site_url']}"
            site_response = requests.get(site_url, headers={'Authorization': f'Bearer {access_token}'})

            if site_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint site"}

            site_data = site_response.json()
            site_id = site_data['id']

            # Get drive
            drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
            drive_response = requests.get(drive_url, headers={'Authorization': f'Bearer {access_token}'})

            if drive_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint drive"}

            drive_data = drive_response.json()
            drive_id = drive_data['id']

            # Create folder path if it doesn't exist
            self.create_sharepoint_folder(site_id, drive_id, folder_path, access_token)

            # Sanitize filename for SharePoint - remove or replace problematic characters
            safe_filename = re.sub(r'[^\w\-_.]', '_', filename)
            
            # URL encode the path components for the API call
            encoded_folder_path = urllib.parse.quote(folder_path, safe='')
            encoded_filename = urllib.parse.quote(safe_filename, safe='')

            # Upload file
            upload_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{encoded_folder_path}/{encoded_filename}:/content"
            print(f"Uploading file to: {upload_url}")

            upload_response = requests.put(upload_url, headers=headers, data=file_content)

            if upload_response.status_code in [200, 201]:
                print(f"✅ Successfully uploaded: {safe_filename}")
                return {"status": "success", "message": f"File uploaded: {safe_filename}"}
            else:
                print(f"❌ Upload failed: {upload_response.status_code} - {upload_response.text}")
                return {"status": "error", "message": f"Upload failed: {upload_response.status_code}"}

        except Exception as e:
            print(f"Error uploading file to SharePoint: {e}")
            return {"status": "error", "message": str(e)}

    def create_sharepoint_folder(self, site_id: str, drive_id: str, folder_path: str, access_token: str):
        """Create folder structure in SharePoint if it doesn't exist"""
        try:
            headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}

            # Split path and create each folder level
            path_parts = folder_path.split('/')
            current_path = ""

            for part in path_parts:
                if part:
                    # Sanitize folder name for SharePoint
                    safe_part = re.sub(r'[^\w\-_.]', '_', part)
                    parent_path = current_path if current_path else ""
                    current_path = f"{current_path}/{safe_part}" if current_path else safe_part

                    # URL encode the path for the API call
                    encoded_current_path = urllib.parse.quote(current_path, safe='')

                    # Check if folder exists
                    check_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{encoded_current_path}"
                    check_response = requests.get(check_url, headers=headers)

                    if check_response.status_code == 404:
                        # Folder doesn't exist, create it
                        if parent_path:
                            encoded_parent_path = urllib.parse.quote(parent_path, safe='')
                            create_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{encoded_parent_path}:/children"
                        else:
                            create_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"

                        folder_data = {
                            "name": safe_part,
                            "folder": {},
                            "@microsoft.graph.conflictBehavior": "rename"
                        }

                        create_response = requests.post(create_url, headers=headers, json=folder_data)
                        if create_response.status_code in [200, 201]:
                            print(f"✅ Created folder: {current_path}")
                        else:
                            print(f"⚠️ Folder creation warning: {create_response.status_code}")

        except Exception as e:
            print(f"Error creating SharePoint folder: {e}")

    def get_file_content(self, path: str, binary: bool = False) -> dict:
        """Get file content from SharePoint"""
        try:
            access_token = self.get_graph_access_token()
            if not access_token:
                return {"status": "error", "message": "Failed to get access token"}

            headers = {'Authorization': f'Bearer {access_token}'}
            
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.graph_config['site_url']}"
            site_response = requests.get(site_url, headers=headers)
            if site_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint site"}
            
            site_data = site_response.json()
            site_id = site_data['id']
            
            drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
            drive_response = requests.get(drive_url, headers=headers)
            if drive_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint drive"}
            
            drive_data = drive_response.json()
            drive_id = drive_data['id']
            
            file_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{path}:/content"
            file_response = requests.get(file_url, headers=headers)
            
            if file_response.status_code == 200:
                content = file_response.content if binary else file_response.text
                return {"status": "success", "content": content}
            else:
                return {"status": "error", "message": f"File not found: {file_response.status_code}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_files(self, path: str = "") -> dict:
        """List files in SharePoint folder"""
        return self.get_folder_contents_by_path(path)

    def delete_file(self, path: str) -> dict:
        """Delete a file from SharePoint"""
        try:
            access_token = self.get_graph_access_token()
            if not access_token:
                return {"status": "error", "message": "Failed to get access token"}

            headers = {'Authorization': f'Bearer {access_token}'}
            
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.graph_config['site_url']}"
            site_response = requests.get(site_url, headers=headers)
            if site_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint site"}
            
            site_data = site_response.json()
            site_id = site_data['id']
            
            drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
            drive_response = requests.get(drive_url, headers=headers)
            if drive_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint drive"}
            
            drive_data = drive_response.json()
            drive_id = drive_data['id']
            
            delete_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{path}"
            delete_response = requests.delete(delete_url, headers=headers)
            
            if delete_response.status_code == 204:
                return {"status": "success", "message": "File deleted successfully"}
            else:
                return {"status": "error", "message": f"Delete failed: {delete_response.status_code}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search_files(self, query: str, path: str = "") -> dict:
        """Search for files in SharePoint"""
        try:
            access_token = self.get_graph_access_token()
            if not access_token:
                return {"status": "error", "message": "Failed to get access token"}

            headers = {'Authorization': f'Bearer {access_token}'}
            
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.graph_config['site_url']}"
            site_response = requests.get(site_url, headers=headers)
            if site_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint site"}
            
            site_data = site_response.json()
            site_id = site_data['id']
            
            drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
            drive_response = requests.get(drive_url, headers=headers)
            if drive_response.status_code != 200:
                return {"status": "error", "message": "Failed to access SharePoint drive"}
            
            drive_data = drive_response.json()
            drive_id = drive_data['id']
            
            search_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/search(q='{query}')"
            search_response = requests.get(search_url, headers=headers)
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                files = []
                for item in search_data.get('value', []):
                    files.append({
                        'id': item['id'],
                        'name': item['name'],
                        'type': 'folder' if 'folder' in item else 'file',
                        'size': item.get('size', 0),
                        'modified': item.get('lastModifiedDateTime', ''),
                        'path': item.get('parentReference', {}).get('path', '') + '/' + item['name']
                    })
                return {"status": "success", "files": files}
            else:
                return {"status": "error", "message": f"Search failed: {search_response.status_code}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def save_link_to_sharepoint(self, link_url: str, link_title: str, folder_path: str):
        """Save a link as a text file to SharePoint"""
        try:
            # Create text file content
            link_content = f"RFP/RFI/RFQ Link\n"
            link_content += f"Title: {link_title}\n"
            link_content += f"URL: {link_url}\n"
            link_content += f"Extracted: {datetime.now().isoformat()}\n"

            # Clean filename
            safe_filename = self.clean_filename(f"{link_title}.txt")

            # Upload as text file
            return self.upload_file_to_sharepoint(
                link_content.encode('utf-8'),
                safe_filename,
                folder_path
            )

        except Exception as e:
            print(f"Error saving link to SharePoint: {e}")
            return {"status": "error", "message": str(e)}

    def clean_filename(self, filename: str) -> str:
        """Clean filename for SharePoint compatibility"""
        import re
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '-', filename)
        # Replace spaces with hyphens
        filename = re.sub(r'\s+', '-', filename)
        # Remove multiple consecutive hyphens
        filename = re.sub(r'-+', '-', filename)
        # Remove leading/trailing hyphens
        filename = filename.strip('-')
        # Limit length
        if len(filename) > 100:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = f"{name[:95]}.{ext}" if ext else name[:100]
        return filename

# SharePoint API Endpoints
@sharepoint_router.get("/test-sharepoint")
async def test_sharepoint():
    """Test SharePoint connection step by step"""
    try:
        print("🧪 Testing SharePoint connection...")

        # Test 1: Access token
        sharepoint_manager = SharePointManager()
        print("📋 Testing access token...")
        token = sharepoint_manager.get_graph_access_token()

        if not token:
            return {"test_result": {"status": "error", "message": "Failed to get access token", "step": "token"}}

        print("✅ Access token obtained")

        # Test 2: Site access
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        site_url = f"https://graph.microsoft.com/v1.0/sites/{sharepoint_manager.graph_config['site_url']}"
        print(f"🌐 Testing site access: {site_url}")

        import requests
        site_response = requests.get(site_url, headers=headers)
        print(f"📊 Site response: {site_response.status_code}")

        if site_response.status_code != 200:
            return {
                "test_result": {
                    "status": "error",
                    "message": f"Site access failed: {site_response.status_code} - {site_response.text[:200]}",
                    "step": "site_access",
                    "site_url": site_url
                }
            }

        print("✅ Site access successful")
        return {"test_result": {"status": "success", "message": "SharePoint connection working", "site_data": site_response.json()}}

    except Exception as e:
        print(f"❌ SharePoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"test_result": {"status": "error", "message": str(e), "step": "exception"}}

@sharepoint_router.get("/files/{folder_path:path}")
async def get_folder_contents(folder_path: str):
    """Get contents of a specific SharePoint folder"""
    try:
        sharepoint_manager = SharePointManager()
        result = sharepoint_manager.get_folder_contents_by_path(folder_path)

        if result['status'] == 'success':
            # Transform SharePoint files to match frontend expectations
            transformed_files = []

            for item in result['files']:
                if item['type'] == 'folder':
                    # Add folder
                    folder_info = {
                        "filename": item['name'],
                        "file_size": 0,
                        "modified_at": item['modified'],
                        "type": "folder",
                        "web_url": item['web_url'],
                        "path": f"{folder_path}/{item['name']}" if folder_path else item['name'],
                        "id": item['id']
                    }
                    transformed_files.append(folder_info)
                else:
                    # Add file
                    file_info = {
                        "filename": item['name'],
                        "file_size": item['size'],
                        "modified_at": item['modified'],
                        "type": item['name'].split('.')[-1].lower() if '.' in item['name'] else 'file',
                        "web_url": item['web_url'],
                        "download_url": item['download_url'],
                        "path": f"{folder_path}/{item['name']}" if folder_path else item['name'],
                        "id": item['id']
                    }
                    transformed_files.append(file_info)

            return {
                "files": transformed_files,
                "status": "success",
                "message": f"Retrieved {len(transformed_files)} items from folder: {folder_path}",
                "current_path": folder_path
            }
        else:
            return {
                "files": [],
                "status": "error",
                "message": result.get('message', 'Failed to get folder contents'),
                "current_path": folder_path
            }

    except Exception as e:
        print(f"Error getting folder contents: {e}")
        return {
            "files": [],
            "status": "error",
            "message": str(e),
            "current_path": folder_path
        }

@sharepoint_router.get("/files")
async def list_files():
    """Get files from SharePoint"""
    try:
        sharepoint_manager = SharePointManager()
        result = sharepoint_manager.get_sharepoint_files()

        if result['status'] == 'success':
            # Transform SharePoint files to match frontend expectations
            transformed_files = []

            def process_items(items, parent_path=""):
                for item in items:
                    if item['type'] == 'folder':
                        # Add folder
                        folder_info = {
                            "filename": item['name'],
                            "file_size": 0,
                            "modified_at": item['modified'],
                            "type": "folder",
                            "web_url": item['web_url'],
                            "path": f"{parent_path}/{item['name']}" if parent_path else item['name'],
                            "children": []
                        }

                        # Process folder contents
                        if 'children' in item:
                            folder_info['children'] = []
                            for child in item['children']:
                                child_info = {
                                    "filename": child['name'],
                                    "file_size": child['size'],
                                    "modified_at": child['modified'],
                                    "type": child['name'].split('.')[-1].lower() if '.' in child['name'] else 'file',
                                    "web_url": child['web_url'],
                                    "download_url": child['download_url'],
                                    "path": f"{folder_info['path']}/{child['name']}"
                                }
                                folder_info['children'].append(child_info)

                        transformed_files.append(folder_info)
                    else:
                        # Add file
                        file_info = {
                            "filename": item['name'],
                            "file_size": item['size'],
                            "modified_at": item['modified'],
                            "type": item['name'].split('.')[-1].lower() if '.' in item['name'] else 'file',
                            "web_url": item['web_url'],
                            "download_url": item['download_url'],
                            "path": f"{parent_path}/{item['name']}" if parent_path else item['name']
                        }
                        transformed_files.append(file_info)

            process_items(result['files'])

            return {
                "files": transformed_files,
                "status": "success",
                "message": f"Retrieved {len(transformed_files)} items from SharePoint"
            }
        else:
            # Fallback to local files if SharePoint fails
            print(f"SharePoint failed: {result.get('message', 'Unknown error')}")
            print("Falling back to local file system...")
            return await get_local_files()

    except Exception as e:
        print(f"Error in list_files: {e}")
        print("Falling back to local file system...")
        return await get_local_files()

async def get_local_files():
    """Fallback to local file system"""
    try:
        import os
        import datetime
        files_dir = os.path.join(os.path.dirname(__file__), "..", "ba-server", "generated_files")
        files = []

        if os.path.exists(files_dir):
            for filename in os.listdir(files_dir):
                file_path = os.path.join(files_dir, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    file_type = filename.split('.')[-1] if '.' in filename else "unknown"

                    files.append({
                        "filename": filename,
                        "file_size": file_size,
                        "modified_at": file_modified.isoformat(),
                        "type": file_type
                    })

        return {"files": files, "status": "success", "message": "Using local files (SharePoint unavailable)"}
    except Exception as e:
        print(f"Local file listing failed: {e}")
        return {"files": [], "status": "error", "message": str(e)}

@sharepoint_router.get("/files/{filename}")
async def download_file(filename: str):
    """Download file from local generated_files directory"""
    from fastapi.responses import FileResponse
    import os
    
    # Path to generated files directory
    file_path = os.path.join(os.path.dirname(__file__), "..", "ba-server", "generated_files", filename)
    
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")

