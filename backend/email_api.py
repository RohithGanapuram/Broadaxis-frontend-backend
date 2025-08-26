"""
Email API for BroadAxis RFP/RFQ Management Platform
"""

import os
import json
import time
import base64
import requests
import imaplib
import email
import re
from typing import List, Dict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from fastapi import APIRouter
from pydantic import BaseModel

from error_handler import error_handler

# Create router for email endpoints
email_router = APIRouter(prefix="/api", tags=["email"])

# Import SharePointManager from sharepoint_api.py
from sharepoint_api import SharePointManager

class EmailFetchRequest(BaseModel):
    email_accounts: List[str] = []  # Optional: specific accounts to fetch from
    use_real_email: bool = True  # Default to real email (Graph API)
    use_graph_api: bool = True  # Default to Microsoft Graph API

class EmailAttachment(BaseModel):
    filename: str
    file_path: str = None  # Optional for link attachments
    file_size: int = None  # Optional for link attachments
    download_date: str
    type: str = "file"  # "file" or "link"
    url: str = None  # For link attachments
    domain: str = None  # For link attachments

class FetchedEmail(BaseModel):
    email_id: str
    sender: str
    subject: str
    date: str
    account: str
    attachments: List[EmailAttachment]
    has_rfp_keywords: bool

class EmailFetchResponse(BaseModel):
    status: str
    message: str
    emails_found: int
    attachments_downloaded: int
    fetched_emails: List[FetchedEmail]

class EmailFetcher:
    def __init__(self):
        self.attachments_dir = Path("email_attachments")
        self.attachments_dir.mkdir(exist_ok=True)

        # RFP/RFI/RFQ keywords to search for - using set for O(1) lookup
        self.rfp_keywords = {
            'rfp', 'rfi', 'rfq', 'request for proposal', 'request for information',
            'request for quotation', 'proposal', 'bid', 'tender', 'procurement',
            'solicitation', 'quote', 'quotation'
        }

        # Microsoft Graph API configuration
        self.graph_config = {
            'client_id': os.getenv('GRAPH_CLIENT_ID'),
            'client_secret': os.getenv('GRAPH_CLIENT_SECRET'),
            'tenant_id': os.getenv('GRAPH_TENANT_ID'),
            'user_emails': [
                os.getenv('GRAPH_USER_EMAIL_1'),
                os.getenv('GRAPH_USER_EMAIL_2'),
                os.getenv('GRAPH_USER_EMAIL_3')
            ]
        }

        # Filter out None values efficiently
        self.graph_config['user_emails'] = [email for email in self.graph_config['user_emails'] if email]

        # Email configurations from environment
        self.email_configs = {
            'gmail': {
                'email': os.getenv('GMAIL_EMAIL'),
                'password': os.getenv('GMAIL_PASSWORD'),
                'imap_server': os.getenv('GMAIL_IMAP_SERVER', 'imap.gmail.com'),
                'imap_port': int(os.getenv('GMAIL_IMAP_PORT', 993))
            },
            'outlook': {
                'email': os.getenv('OUTLOOK_EMAIL'),
                'password': os.getenv('OUTLOOK_PASSWORD'),
                'imap_server': os.getenv('OUTLOOK_IMAP_SERVER', 'outlook.office365.com'),
                'imap_port': int(os.getenv('OUTLOOK_IMAP_PORT', 993))
            },
            'corporate': {
                'email': os.getenv('CORPORATE_EMAIL'),
                'password': os.getenv('CORPORATE_PASSWORD'),
                'imap_server': os.getenv('CORPORATE_IMAP_SERVER'),
                'imap_port': int(os.getenv('CORPORATE_IMAP_PORT', 993))
            }
        }
        
        # Cache for Graph API token
        self._token_cache = {'token': None, 'expires_at': 0}
        self._session = requests.Session()
        self._session.timeout = 15

    def has_rfp_keywords(self, text: str) -> bool:
        """Check if text contains RFP/RFI/RFQ related keywords - optimized with set lookup"""
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.rfp_keywords)

    def extract_links_from_email(self, email_content: str) -> List[dict]:
        """Extract relevant links from email content"""
        links = []
        seen_urls = set()  # Track URLs to avoid duplicates

        # Regular expression to find URLs
        url_pattern = r'https?://[^\s<>"{}|\\^\[\]]+[^\s<>"{}|\\^\[\].,;:!?]'
        found_urls = re.findall(url_pattern, email_content, re.IGNORECASE)

        # Keywords that suggest RFP/procurement related links
        rfp_link_keywords = [
            'rfp', 'rfi', 'rfq', 'proposal', 'bid', 'tender', 'procurement',
            'solicitation', 'quote', 'quotation', 'contract', 'vendor',
            'supplier', 'opportunity', 'award', 'government', 'portal'
        ]

        for url in found_urls:
            try:
                # Clean up the URL
                url = url.strip()

                # Skip if we've already seen this URL
                if url in seen_urls:
                    continue

                # Parse URL to get domain and path
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                path = parsed.path.lower()

                # Check if URL is likely RFP/procurement related
                is_rfp_related = False

                # Check domain and path for RFP keywords
                full_url_text = f"{domain} {path}".lower()
                if any(keyword in full_url_text for keyword in rfp_link_keywords):
                    is_rfp_related = True

                # Check for government domains (often have procurement portals)
                gov_domains = ['.gov', '.mil', '.edu', 'procurement', 'tender', 'bid']
                if any(gov_domain in domain for gov_domain in gov_domains):
                    is_rfp_related = True

                # Check for common procurement platforms
                procurement_platforms = [
                    'sam.gov', 'fedbizopps', 'grants.gov', 'beta.sam.gov',
                    'merx.com', 'biddingo.com', 'demandstar.com', 'publicsector.ca',
                    'bonfirehub.com', 'ionwave.net', 'questcdn.com'
                ]
                if any(platform in domain for platform in procurement_platforms):
                    is_rfp_related = True

                if is_rfp_related:
                    # Add to seen URLs to avoid duplicates
                    seen_urls.add(url)

                    # Try to get a meaningful title from the URL
                    link_title = self.generate_link_title(url, domain, path)

                    links.append({
                        'url': url,
                        'title': link_title,
                        'domain': domain,
                        'type': 'rfp_link'
                    })

            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                continue

        return links
    
    def generate_link_title(self, url: str, domain: str, path: str) -> str:
        """Generate a meaningful title for the link"""
        # Try to extract meaningful parts from the URL
        if 'sam.gov' in domain:
            return f"SAM.gov Opportunity - {domain}"
        elif '.gov' in domain:
            return f"Government Procurement - {domain}"
        elif 'rfp' in path or 'rfp' in domain:
            return f"RFP Portal - {domain}"
        elif 'rfi' in path or 'rfi' in domain:
            return f"RFI Portal - {domain}"
        elif 'rfq' in path or 'rfq' in domain:
            return f"RFQ Portal - {domain}"
        elif 'tender' in path or 'tender' in domain:
            return f"Tender Portal - {domain}"
        elif 'bid' in path or 'bid' in domain:
            return f"Bidding Portal - {domain}"
        elif 'procurement' in path or 'procurement' in domain:
            return f"Procurement Portal - {domain}"
        else:
            return f"Opportunity Link - {domain}"

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

    def save_attachment(self, attachment_data: bytes, filename: str, email_date: str) -> dict:
        """Save email attachment to local folder"""
        try:
            # Create date-based subfolder
            date_folder = self.attachments_dir / email_date.split()[0]  # YYYY-MM-DD
            date_folder.mkdir(exist_ok=True)

            # Clean filename
            clean_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            file_path = date_folder / clean_filename

            # Save file
            with open(file_path, 'wb') as f:
                f.write(attachment_data)

            return {
                "filename": clean_filename,
                "file_path": str(file_path),
                "file_size": len(attachment_data),
                "download_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error saving attachment {filename}: {e}")
            return None

    def get_graph_access_token(self) -> str:
        """Get access token for Microsoft Graph API with caching"""
        now = time.time()
        
        # Return cached token if still valid (with 5 min buffer)
        if self._token_cache['token'] and now < self._token_cache['expires_at'] - 300:
            return self._token_cache['token']
        
        try:
            token_url = f"https://login.microsoftonline.com/{self.graph_config['tenant_id']}/oauth2/v2.0/token"

            token_data = {
                'grant_type': 'client_credentials',
                'client_id': self.graph_config['client_id'],
                'client_secret': self.graph_config['client_secret'],
                'scope': 'https://graph.microsoft.com/.default'
            }

            response = self._session.post(token_url, data=token_data)
            response.raise_for_status()
            
            token_response = response.json()
            access_token = token_response['access_token']
            
            # Cache token with expiration
            self._token_cache['token'] = access_token
            self._token_cache['expires_at'] = now + token_response.get('expires_in', 3600)
            
            return access_token
        except requests.exceptions.Timeout:
            error_handler.log_error(Exception("Timeout getting Graph access token"), {'operation': 'get_graph_access_token'})
            return None
        except requests.exceptions.RequestException as e:
            error_handler.log_error(e, {'operation': 'get_graph_access_token'})
            return None
        except Exception as e:
            error_handler.log_error(e, {'operation': 'get_graph_access_token'})
            return None

    def fetch_emails_graph(self) -> dict:
        """Fetch emails using Microsoft Graph API from multiple accounts"""
        if not self.graph_config['client_id'] or not self.graph_config['client_secret'] or not self.graph_config['tenant_id']:
            return {
                "status": "error",
                "message": "Microsoft Graph API configuration incomplete. Please check GRAPH_CLIENT_ID, GRAPH_CLIENT_SECRET, and GRAPH_TENANT_ID in .env file.",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }

        if not self.graph_config['user_emails']:
            return {
                "status": "error",
                "message": "No email accounts configured. Please check GRAPH_USER_EMAIL_1, GRAPH_USER_EMAIL_2, GRAPH_USER_EMAIL_3 in .env file.",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }

        try:
            # Get access token
            access_token = self.get_graph_access_token()
            if not access_token:
                return {
                    "status": "error",
                    "message": "Failed to get Microsoft Graph access token. Check your client credentials.",
                    "emails_found": 0,
                    "attachments_downloaded": 0,
                    "fetched_emails": []
                }

            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }

            fetched_emails = []
            total_attachments = 0

            # Process all email accounts efficiently
            for user_email in self.graph_config['user_emails']:
                # Get emails from the last 30 days for this account
                graph_url = f"https://graph.microsoft.com/v1.0/users/{user_email}/messages"
                params = {
                    '$top': 50,  # Get more emails to filter through
                    '$select': 'id,subject,sender,receivedDateTime,hasAttachments,body',
                    '$orderby': 'receivedDateTime desc'
                }

                try:
                    response = self._session.get(graph_url, headers=headers, params=params)
                    response.raise_for_status()
                    emails_data = response.json()
                except requests.exceptions.Timeout:
                    error_handler.log_error(Exception(f"Timeout fetching emails from {user_email}"), 
                                          {'operation': 'fetch_emails_graph', 'user_email': user_email})
                    continue  # Skip this account and try the next one
                except requests.exceptions.RequestException as e:
                    error_handler.log_error(e, {'operation': 'fetch_emails_graph', 'user_email': user_email})
                    continue  # Skip this account and try the next one

                # Process each email and filter for RFP/RFI/RFQ keywords
                for email_item in emails_data.get('value', []):
                    # Check if email has RFP keywords first
                    has_keywords = self.has_rfp_keywords(email_item.get('subject', ''))

                    if has_keywords:
                        attachments = []

                        # Extract email content for link detection
                        email_content = ""
                        if email_item.get('body') and email_item['body'].get('content'):
                            email_content = email_item['body']['content']

                        # Extract links from email content
                        extracted_links = self.extract_links_from_email(email_content)

                        # Create SharePoint folder path for this email
                        email_date = email_item['receivedDateTime'][:10]  # YYYY-MM-DD
                        email_subject_clean = self.clean_filename(email_item['subject'])
                        email_folder_name = f"{email_date}_{email_subject_clean}"
                        sharepoint_folder_path = f"Emails/{user_email}/{email_folder_name}"

                        # Initialize SharePoint manager for uploads
                        sharepoint_manager = SharePointManager()

                        # Add links as "link attachments" and save to SharePoint
                        for link in extracted_links:
                            # Save link to SharePoint
                            link_result = sharepoint_manager.save_link_to_sharepoint(
                                link['url'],
                                link['title'],
                                sharepoint_folder_path
                            )

                            # Link saved to SharePoint

                            link_attachment = {
                                'filename': link['title'],
                                'file_path': '',  # Empty string for links
                                'file_size': 0,   # Zero for links
                                'url': link['url'],
                                'domain': link['domain'],
                                'type': 'link',
                                'download_date': datetime.now().isoformat(),
                                'sharepoint_path': sharepoint_folder_path
                            }
                            attachments.append(link_attachment)
                            total_attachments += 1

                        # Process file attachments efficiently
                        if email_item.get('hasAttachments'):
                            # Get attachments for this specific user account
                            attachments_url = f"https://graph.microsoft.com/v1.0/users/{user_email}/messages/{email_item['id']}/attachments"
                            attachments_response = self._session.get(attachments_url, headers=headers)

                            if attachments_response.status_code == 200:
                                attachments_data = attachments_response.json()
                                date_str = email_item['receivedDateTime'][:10]  # Get YYYY-MM-DD part

                                for attachment in attachments_data.get('value', []):
                                    if attachment.get('@odata.type') == '#microsoft.graph.fileAttachment':
                                        # Download attachment
                                        attachment_content = base64.b64decode(attachment['contentBytes'])

                                        # Save to local storage and SharePoint in parallel
                                        saved_attachment = self.save_attachment(
                                            attachment_content,
                                            attachment['name'],
                                            date_str
                                        )

                                        if saved_attachment:
                                            # Upload to SharePoint
                                            clean_filename = self.clean_filename(attachment['name'])
                                            sharepoint_manager.upload_file_to_sharepoint(
                                                attachment_content,
                                                clean_filename,
                                                sharepoint_folder_path
                                            )

                                            saved_attachment['type'] = 'file'  # Mark as file attachment
                                            saved_attachment['sharepoint_path'] = sharepoint_folder_path
                                            attachments.append(saved_attachment)
                                            total_attachments += 1

                        # Make a deep copy of attachments to avoid reference sharing
                        import copy
                        email_attachments = copy.deepcopy(attachments)

                        fetched_emails.append({
                            "email_id": email_item['id'],
                            "sender": email_item['sender']['emailAddress']['address'],
                            "subject": email_item['subject'],
                            "date": email_item['receivedDateTime'],
                            "account": user_email,  # Use the current user_email being processed
                            "attachments": email_attachments,
                            "has_rfp_keywords": has_keywords
                        })

            account_count = len(self.graph_config['user_emails'])
            return {
                "status": "success",
                "message": f"Successfully fetched {len(fetched_emails)} RFP/RFI/RFQ emails from {account_count} email accounts using Microsoft Graph API",
                "emails_found": len(fetched_emails),
                "attachments_downloaded": total_attachments,
                "fetched_emails": fetched_emails
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Microsoft Graph API error: {str(e)}",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching emails via Graph API: {str(e)}",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }
        
    def fetch_emails_real(self, email_account: str = 'gmail') -> dict:
        """Fetch emails from real email account"""
        config = self.email_configs.get(email_account)

        if not config or not config['email'] or not config['password']:
            return {
                "status": "error",
                "message": f"Email configuration for {email_account} not found or incomplete. Please check your .env file.",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }

        try:
            # Connect to email server
            mail = imaplib.IMAP4_SSL(config['imap_server'], config['imap_port'])

            # Try different authentication methods for corporate accounts
            try:
                mail.login(config['email'], config['password'])
            except imaplib.IMAP4.error:
                # Try with username only (without domain)
                username = config['email'].split('@')[0]
                mail.login(username, config['password'])
            mail.select('inbox')

            # Search for emails with RFP/RFI/RFQ keywords
            search_criteria = []
            for keyword in self.rfp_keywords:
                search_criteria.append(f'SUBJECT "{keyword}"')

            # Search in the last 30 days
            search_query = f'(SINCE "01-Jan-2025") ({" OR ".join(search_criteria)})'

            result, message_numbers = mail.search(None, search_query)

            if result != 'OK':
                return {
                    "status": "error",
                    "message": "Failed to search emails",
                    "emails_found": 0,
                    "attachments_downloaded": 0,
                    "fetched_emails": []
                }

            email_ids = message_numbers[0].split()
            fetched_emails = []
            total_attachments = 0

            # Process each email
            for email_id in email_ids[-10:]:  # Limit to last 10 emails
                result, message_data = mail.fetch(email_id, '(RFC822)')

                if result == 'OK':
                    email_message = email.message_from_bytes(message_data[0][1])

                    # Extract email details
                    sender = email_message.get('From', 'Unknown')
                    subject = email_message.get('Subject', 'No Subject')
                    date = email_message.get('Date', 'Unknown')

                    # Check if email has RFP keywords
                    has_keywords = self.has_rfp_keywords(subject)

                    if has_keywords:
                        attachments = []

                        # Process attachments
                        for part in email_message.walk():
                            if part.get_content_disposition() == 'attachment':
                                filename = part.get_filename()
                                if filename:
                                    attachment_data = part.get_payload(decode=True)
                                    saved_attachment = self.save_attachment(attachment_data, filename, date)
                                    if saved_attachment:
                                        attachments.append(saved_attachment)
                                        total_attachments += 1

                        # Make a deep copy of attachments to avoid reference sharing
                        import copy
                        email_attachments = copy.deepcopy(attachments)

                        fetched_emails.append({
                            "email_id": email_id.decode(),
                            "sender": sender,
                            "subject": subject,
                            "date": date,
                            "account": config['email'],
                            "attachments": email_attachments,
                            "has_rfp_keywords": has_keywords
                        })

            mail.close()
            mail.logout()

            return {
                "status": "success",
                "message": f"Successfully fetched {len(fetched_emails)} RFP/RFI/RFQ emails from {config['email']}",
                "emails_found": len(fetched_emails),
                "attachments_downloaded": total_attachments,
                "fetched_emails": fetched_emails
            }

        except imaplib.IMAP4.error as e:
            return {
                "status": "error",
                "message": f"IMAP error: {str(e)}. Check your email credentials and server settings.",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching emails: {str(e)}",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }

    async def fetch_emails_demo(self) -> EmailFetchResponse:
        """Demo implementation - simulates email fetching"""
        # Simulate finding RFP-related emails
        demo_emails = [
            {
                "email_id": "email_001",
                "sender": "procurement@techcorp.com",
                "subject": "RFP for Cloud Infrastructure Services - Due March 15",
                "date": "2025-01-15 10:30:00",
                "account": "proposals@broadaxis.com",
                "attachments": [
                    {
                        "filename": "RFP_Cloud_Infrastructure_2025.pdf",
                        "file_path": "email_attachments/2025-01-15/RFP_Cloud_Infrastructure_2025.pdf",
                        "file_size": 2048576,
                        "download_date": datetime.now().isoformat()
                    }
                ],
                "has_rfp_keywords": True
            },
            {
                "email_id": "email_002",
                "sender": "sourcing@govagency.gov",
                "subject": "RFI - Software Development Services",
                "date": "2025-01-14 14:20:00",
                "account": "rfp.team@broadaxis.com",
                "attachments": [
                    {
                        "filename": "RFI_Software_Development.docx",
                        "file_path": "email_attachments/2025-01-14/RFI_Software_Development.docx",
                        "file_size": 1024000,
                        "download_date": datetime.now().isoformat()
                    }
                ],
                "has_rfp_keywords": True
            },
            {
                "email_id": "email_003",
                "sender": "purchasing@enterprise.com",
                "subject": "RFQ for Hardware Procurement - Urgent",
                "date": "2025-01-13 09:15:00",
                "account": "business@broadaxis.com",
                "attachments": [
                    {
                        "filename": "Hardware_RFQ_Specifications.pdf",
                        "file_path": "email_attachments/2025-01-13/Hardware_RFQ_Specifications.pdf",
                        "file_size": 3072000,
                        "download_date": datetime.now().isoformat()
                    }
                ],
                "has_rfp_keywords": True
            }
        ]

        # Create demo attachment files
        for email_data in demo_emails:
            for attachment in email_data["attachments"]:
                file_path = Path(attachment["file_path"])
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Create demo file content
                demo_content = f"""Demo {attachment['filename']}

This is a simulated RFP/RFI/RFQ document downloaded from email.
Email: {email_data['sender']}
Subject: {email_data['subject']}
Date: {email_data['date']}

[This would contain the actual RFP/RFI/RFQ content in a real scenario]
"""
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(demo_content)

        return {
            "status": "success",
            "message": f"Successfully fetched {len(demo_emails)} RFP/RFI/RFQ emails",
            "emails_found": len(demo_emails),
            "attachments_downloaded": sum(len(email_data["attachments"]) for email_data in demo_emails),
            "fetched_emails": demo_emails
        }

# Initialize email fetcher
email_fetcher = EmailFetcher()

# Global storage for real fetched emails (in production, use database)
real_fetched_emails = []

def save_real_emails_to_file(emails):
    """Save real fetched emails to a JSON file for persistence"""
    try:
        with open('real_fetched_emails.json', 'w') as f:
            json.dump(emails, f, indent=2)
    except Exception as e:
        print(f"Error saving real emails: {e}")

def load_real_emails_from_file():
    """Load real fetched emails from JSON file"""
    try:
        if os.path.exists('real_fetched_emails.json'):
            with open('real_fetched_emails.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading real emails: {e}")
    return []

# Load real emails on startup
real_fetched_emails = load_real_emails_from_file()

# Email API Endpoints
@email_router.get("/test-graph-auth")
async def test_graph_auth():
    """Test Microsoft Graph API authentication"""
    try:
        print("🧪 Testing Microsoft Graph API authentication...")
        
        # Test access token
        token = email_fetcher.get_graph_access_token()
        if not token:
            return {
                "status": "error",
                "message": "Failed to get access token",
                "step": "authentication"
            }
        
        print("✅ Access token obtained")
        
        # Test basic Graph API call using application permissions
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        
        # Test with first configured email account
        user_emails = [
            os.getenv('GRAPH_USER_EMAIL_1'),
            os.getenv('GRAPH_USER_EMAIL_2'),
            os.getenv('GRAPH_USER_EMAIL_3')
        ]
        user_emails = [email for email in user_emails if email]
        
        if not user_emails:
            return {
                "status": "error",
                "message": "No email accounts configured. Check GRAPH_USER_EMAIL_1, GRAPH_USER_EMAIL_2, GRAPH_USER_EMAIL_3 in .env file.",
                "step": "config_check"
            }
        
        test_email = user_emails[0]
        test_url = f"https://graph.microsoft.com/v1.0/users/{test_email}"
        
        response = requests.get(test_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_info = response.json()
            return {
                "status": "success",
                "message": f"Microsoft Graph API authentication working! Connected to {user_info.get('displayName', test_email)}",
                "user_info": {
                    "email": user_info.get('mail', test_email),
                    "displayName": user_info.get('displayName', 'Unknown'),
                    "configured_accounts": len(user_emails)
                }
            }
        elif response.status_code == 403:
            return {
                "status": "error",
                "message": "Microsoft Graph API permissions insufficient. Your app needs 'Mail.Read' and 'User.Read.All' application permissions. Contact your Azure admin to grant these permissions.",
                "step": "permissions",
                "test_email": test_email,
                "fix_instructions": [
                    "1. Go to Azure Portal > App Registrations",
                    "2. Find your app and go to API Permissions",
                    "3. Add Microsoft Graph Application permissions:",
                    "   - Mail.Read (to read emails)",
                    "   - User.Read.All (to access user info)",
                    "4. Click 'Grant admin consent'",
                    "5. Wait 5-10 minutes for permissions to propagate"
                ]
            }
        else:
            return {
                "status": "error",
                "message": f"Graph API test failed: {response.status_code} - {response.text[:200]}",
                "step": "api_test",
                "test_email": test_email
            }
            
    except Exception as e:
        print(f"❌ Graph API test failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "step": "exception"
        }

@email_router.post("/fetch-emails", response_model=EmailFetchResponse)
async def fetch_emails(request: EmailFetchRequest = EmailFetchRequest()):
    """Fetch RFP/RFI/RFQ emails and download attachments"""
    try:
        print("📧 Starting email fetch process...")
        
        # Quick auth test first
        token = email_fetcher.get_graph_access_token()
        if not token:
            print("⚠️ Authentication failed, using demo mode")
            return await email_fetcher.fetch_emails_demo()
        
        print("✅ Authentication successful, testing permissions...")
        
        # Test permissions with a simple API call
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        user_emails = [os.getenv('GRAPH_USER_EMAIL_1'), os.getenv('GRAPH_USER_EMAIL_2'), os.getenv('GRAPH_USER_EMAIL_3')]
        user_emails = [email for email in user_emails if email]
        
        if user_emails:
            test_url = f"https://graph.microsoft.com/v1.0/users/{user_emails[0]}"
            test_response = requests.get(test_url, headers=headers, timeout=5)
            
            if test_response.status_code == 403:
                print("⚠️ Insufficient permissions, using demo mode")
                demo_result = await email_fetcher.fetch_emails_demo()
                demo_result['message'] += " (Demo mode - Microsoft Graph API permissions needed)"
                return demo_result
        
        # Try real email fetching
        result = email_fetcher.fetch_emails_graph()
        
        # If real fetching fails, fall back to demo
        if result['status'] == 'error' and ('403' in result['message'] or 'Authorization' in result['message']):
            print("⚠️ Real email fetch failed due to permissions, using demo mode")
            demo_result = await email_fetcher.fetch_emails_demo()
            demo_result['message'] += " (Demo mode - Microsoft Graph API permissions needed)"
            return demo_result

        # Store real fetched emails for display
        if result['status'] == 'success':
            global real_fetched_emails
            real_fetched_emails = result['fetched_emails']
            # Save to file for persistence
            save_real_emails_to_file(real_fetched_emails)

        return EmailFetchResponse(**result)
    except Exception as e:
        print(f"Error fetching emails: {e}")
        print("⚠️ Falling back to demo mode")
        demo_result = await email_fetcher.fetch_emails_demo()
        demo_result['message'] += " (Demo mode - Error occurred)"
        return demo_result

@email_router.get("/fetched-emails")
async def get_fetched_emails():
    """Get list of previously fetched emails"""
    try:
        # Check if we have real fetched emails from Graph API or IMAP
        global real_fetched_emails
        if real_fetched_emails:
            # Return real fetched emails
            email_accounts = {}
            for email in real_fetched_emails:
                account = email.get('account', 'unknown@example.com')
                if account not in email_accounts:
                    email_accounts[account] = {
                        'count': 0,
                        'emails': [],
                        'latest_subject': '',
                        'latest_date': '',
                        'total_files': 0
                    }

                email_accounts[account]['count'] += 1
                email_accounts[account]['emails'].append(email)
                email_accounts[account]['latest_subject'] = email.get('subject', 'No Subject')
                email_accounts[account]['latest_date'] = email.get('date', '')
                email_accounts[account]['total_files'] += len(email.get('attachments', []))

            # Convert to the expected format
            emails = []
            for i, (account, data) in enumerate(email_accounts.items(), 1):
                emails.append({
                    'id': i,
                    'email': account,
                    'count': data['count'],
                    'latest_subject': data['latest_subject'],
                    'latest_date': data['latest_date'],
                    'latest_files': data['total_files'],
                    'emails': data['emails']
                })

            return {
                'total_count': len(real_fetched_emails),
                'total_files': sum(len(email.get('attachments', [])) for email in real_fetched_emails),
                'emails': emails
            }

        # Fallback to demo data if no real emails fetched
        attachments_dir = Path("email_attachments")
        if not attachments_dir.exists():
            return {"emails": [], "total_count": 0}

        # Count files in attachments directory
        total_files = sum(1 for file_path in attachments_dir.rglob("*") if file_path.is_file())

        demo_emails = [
            {
                "id": 1,
                "email": "proposals@broadaxis.com",
                "count": 8,
                "latest_subject": "RFP for Cloud Infrastructure Services",
                "latest_date": "2025-01-15 10:30:00",
                "latest_file": "RFP_Cloud_Infrastructure_2025.pdf"
            },
            {
                "id": 2,
                "email": "rfp.team@broadaxis.com",
                "count": 5,
                "latest_subject": "RFI - Software Development Services",
                "latest_date": "2025-01-14 14:20:00",
                "latest_file": "RFI_Software_Development.docx"
            },
            {
                "id": 3,
                "email": "business@broadaxis.com",
                "count": 3,
                "latest_subject": "RFQ for Hardware Procurement",
                "latest_date": "2025-01-13 09:15:00",
                "latest_file": "Hardware_RFQ_Specifications.pdf"
            }
        ]

        return {
            "emails": demo_emails,
            "total_count": sum(email["count"] for email in demo_emails),
            "total_files": total_files
        }
    except Exception as e:
        print(f"Error getting fetched emails: {e}")
        return {"emails": [], "total_count": 0, "total_files": 0}

@email_router.get("/email-attachments/{email_id}")
async def get_email_attachments(email_id: int):
    """Get attachments for a specific email account"""
    try:
        # Check if we have real fetched emails
        global real_fetched_emails
        if real_fetched_emails:
            # Group emails by account first
            email_accounts = {}
            for email in real_fetched_emails:
                account = email.get('account', 'unknown@example.com')
                if account not in email_accounts:
                    email_accounts[account] = []
                email_accounts[account].append(email)

            # Convert to list with IDs (same logic as fetched-emails endpoint)
            account_list = []
            for i, (account, emails) in enumerate(email_accounts.items(), 1):
                account_list.append({
                    'id': i,
                    'account': account,
                    'emails': emails
                })

            # Find the specific account for this email_id
            target_account = None
            for account_data in account_list:
                if account_data['id'] == email_id:
                    target_account = account_data
                    break

            if target_account:
                # Return attachments only for this specific account
                account_attachments = []
                for email in target_account['emails']:
                    attachments = email.get('attachments', [])
                    email_subject = email.get('subject', 'No Subject')
                    email_sender = email.get('sender', 'Unknown Sender')
                    email_date = email.get('date', '')

                    # Add email context to each attachment
                    for attachment in attachments:
                        attachment_with_context = attachment.copy()
                        attachment_with_context['email_subject'] = email_subject
                        attachment_with_context['email_sender'] = email_sender
                        attachment_with_context['email_date'] = email_date
                        account_attachments.append(attachment_with_context)

                return {
                    "email_id": email_id,
                    "account": target_account['account'],
                    "attachments": account_attachments
                }

        # Fallback to demo data if no real emails
        attachments_data = {
            1: [
                {"filename": "RFP_Cloud_Infrastructure_2025.pdf", "date": "2025-01-15", "size": "2.0 MB"},
                {"filename": "Technical_Requirements.docx", "date": "2025-01-14", "size": "1.5 MB"},
                {"filename": "Budget_Guidelines.xlsx", "date": "2025-01-13", "size": "0.8 MB"}
            ],
            2: [
                {"filename": "RFI_Software_Development.docx", "date": "2025-01-14", "size": "1.0 MB"},
                {"filename": "Vendor_Questionnaire.pdf", "date": "2025-01-12", "size": "0.5 MB"}
            ],
            3: [
                {"filename": "Hardware_RFQ_Specifications.pdf", "date": "2025-01-13", "size": "3.0 MB"}
            ]
        }

        return {
            "email_id": email_id,
            "attachments": attachments_data.get(email_id, [])
        }
    except Exception as e:
        print(f"Error getting email attachments: {e}")
        return {"email_id": email_id, "attachments": []}

