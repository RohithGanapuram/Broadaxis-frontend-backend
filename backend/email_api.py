"""
Email API for BroadAxis RFP/RFQ Management Platform
"""

import os
import json
import time
import base64
import requests
from pydantic import BaseModel
from typing import Optional, Literal

import re
from typing import List, Dict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from fastapi import APIRouter
from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
# CORS middleware removed - handled by main api.py
from fastapi.exception_handlers import http_exception_handler
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ValidationError as PydanticValidationError
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

from error_handler import error_handler

# Create router for email endpoints
email_router = APIRouter(prefix="/api", tags=["email"])

# Import SharePointManager from sharepoint_api.py
from sharepoint_api import SharePointManager


import json
from pathlib import Path
from datetime import datetime, timedelta

PROCESSED_INDEX_PATH = Path("processed_attachments.json")

def load_processed_index() -> dict[str, set]:
    try:
        raw = json.loads(PROCESSED_INDEX_PATH.read_text(encoding="utf-8"))
        # convert lists -> sets
        return {acct: set(items) for acct, items in raw.items()}
    except Exception:
        return {}

def save_processed_index(idx: dict[str, set]) -> None:
    # convert sets -> lists for json
    clean = {acct: sorted(list(items)) for acct, items in idx.items()}
    PROCESSED_INDEX_PATH.write_text(json.dumps(clean, indent=2), encoding="utf-8")
def _norm_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    # simple normalizer: lowercase scheme+host, trim trailing slash
    try:
        from urllib.parse import urlsplit, urlunsplit
        sp = urlsplit(u)
        host = (sp.hostname or "").lower()
        scheme = (sp.scheme or "").lower()
        rebuilt = urlunsplit((scheme, host + (f":{sp.port}" if sp.port else ""), sp.path or "", sp.query or "", sp.fragment or ""))
        return rebuilt.rstrip("/")
    except Exception:
        return u.rstrip("/")




REAL_EMAILS_PATH = Path("real_fetched_emails.json")

def load_real_emails_from_file() -> list:
    try:
        if REAL_EMAILS_PATH.exists():
            return json.loads(REAL_EMAILS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

def save_real_emails_to_file(data: list) -> None:
    try:
        REAL_EMAILS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

def _att_key(att: dict) -> str:
    t = (att.get('type') or 'file').lower()

    # Links: normalize URL so the same link always hashes to the same key
    if t == 'link':
        return f"link|{_norm_url(att.get('url', ''))}"

    # Files: prefer sharepoint_path + filename (most stable across runs)
    name = (att.get('filename') or '').strip().lower()
    sp_path = (att.get('sharepoint_path') or '').strip().lower()

    # coerce size to int if present; may be missing or a string
    size_val = att.get('file_size')
    try:
        size = int(size_val) if size_val is not None and str(size_val).strip() != '' else None
    except Exception:
        size = None

    if sp_path:
        # same folder + same filename => same file
        return f"file|{sp_path}|{name}"
    if size is not None:
        # fallback when we don't have sharepoint_path
        return f"file|{name}|{size}"
    # last resort: name only
    return f"file|{name}"
from pathlib import Path

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.tif', '.tiff'}

def _is_image_attachment(att: dict) -> bool:
    # Only screen file-type attachments; leave links alone
    if (att.get('type') or 'file').lower() != 'file':
        return False
    name = (att.get('filename') or att.get('file_path') or '').strip()
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXTS:
        return True
    # also respect a content type field if present
    ctype = (att.get('content_type') or att.get('contentType') or '')
    return isinstance(ctype, str) and ctype.lower().startswith('image/')


def _merge_email_batches(existing: list, new: list) -> list:
    by_key = {(e.get('account'), e.get('email_id')): e for e in existing}
    for e in new:
        k = (e.get('account'), e.get('email_id'))
        if k not in by_key:
            by_key[k] = e
            continue
        curr = by_key[k]
        curr_atts = curr.get('attachments', [])
        seen_keys = {_att_key(a) for a in curr_atts}
        for a in e.get('attachments', []):
            ak = _att_key(a)
            if ak in seen_keys:
                continue
            curr_atts.append(a)
            seen_keys.add(ak)
        curr['attachments'] = curr_atts
    return list(by_key.values())

def _get_cache_emails() -> list:
    # unified cache reader for GET endpoints
    global real_fetched_emails
    return real_fetched_emails if real_fetched_emails else load_real_emails_from_file()

def _dedup_attachments(att_list: list[dict]) -> list[dict]:
    out, seen = [], set()
    for a in att_list or []:
        k = _att_key(a)
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out

from datetime import datetime, timezone

def _parse_dt(s: str) -> datetime:
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        # handle Graph's 'Z'
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

def _ordered_accounts(emails):
    """
    Group emails by account and return a list of (account, emails_list)
    sorted by account for stable IDs.
    """
    per_acct = {}
    for em in emails or []:
        acct = em.get('account', 'unknown@example.com')
        per_acct.setdefault(acct, []).append(em)

    # stable order (alphabetical, case-insensitive)
    ordered = []
    for acct in sorted(per_acct.keys(), key=lambda x: (x or '').lower()):
        ordered.append((acct, per_acct[acct]))
    return ordered

app = FastAPI(title="BroadAxis API", version="1.0.0")

# CORS is handled by the main api.py app

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
    
    sharepoint_path: Optional[str] = None
    sharepoint_web_url: Optional[str] = None
    sharepoint_download_url: Optional[str] = None

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

        # RFP/RFI/RFQ keywords to search for
        self.rfp_keywords = [
            'rfp', 'rfi', 'rfq', 'request for proposal', 'request for information',
            'request for quotation', 'proposal', 'bid', 'tender', 'procurement',
            'solicitation', 'quote', 'quotation'
        ]

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

        # Filter out None values
        self.graph_config['user_emails'] = [email for email in self.graph_config['user_emails'] if email]

        # Email configurations from environment
        
    

    def has_rfp_keywords(self, text: str) -> bool:
        """Check if text contains RFP/RFI/RFQ related keywords"""
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



    def _attachments_match_keywords(self, user_email: str, message_id: str, headers: dict) -> bool:
   

        try:
            att_url = f"https://graph.microsoft.com/v1.0/users/{user_email}/messages/{message_id}/attachments"
            # Don't over-select; some fields differ by type. We only need "name".
            r = requests.get(att_url, headers=headers, timeout=10)
            r.raise_for_status()
            for a in (r.json().get('value') or []):
                name = (a.get('name') or '')
                if name and self.has_rfp_keywords(name):
                    return True
            return False
        except Exception:
            return False


    def get_graph_access_token(self) -> str:
        """Get access token for Microsoft Graph API"""
        try:
            token_url = f"https://login.microsoftonline.com/{self.graph_config['tenant_id']}/oauth2/v2.0/token"

            token_data = {
                'grant_type': 'client_credentials',
                'client_id': self.graph_config['client_id'],
                'client_secret': self.graph_config['client_secret'],
                'scope': 'https://graph.microsoft.com/.default'
            }

            response = requests.post(token_url, data=token_data, timeout=10)
            response.raise_for_status()

            return response.json()['access_token']
        except requests.exceptions.Timeout:
            print("Timeout getting Graph access token")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error getting Graph access token: {e}")
            return None
        except Exception as e:
            print(f"Error getting Graph access token: {e}")
            return None

    def fetch_emails_graph(self) -> dict:
   
    # Config checks
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
            # Token
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

            # Last 90 days (tz-aware UTC)
            cutoff = datetime.now(timezone.utc) - timedelta(days=3)
            cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")  # Graph $filter needs Z

            # De-dupe state + existing UI cache
            processed = load_processed_index() or {}  # { mailbox: set(keys) }
            existing_cache = _get_cache_emails() or []  # your persisted UI cache
            existing_map = {
                (e.get('account'), e.get('email_id')): (e.get('attachments') or [])
                for e in existing_cache
            }

            batch_all = []                # ALL matching emails for UI
            total_new_attachments = 0
            total_scanned_emails = 0

            for user_email in self.graph_config['user_emails']:
                seen = processed.setdefault(user_email, set())

                url = f"https://graph.microsoft.com/v1.0/users/{user_email}/mailFolders/Inbox/messages"
                params = {
                    '$filter': f"receivedDateTime ge {cutoff_iso}",
                    '$orderby': 'receivedDateTime desc',
                    '$select': 'id,subject,sender,receivedDateTime,hasAttachments,body',
                    '$top': 50
                }

                while True:
                    resp = requests.get(url, headers=headers, params=params, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()

                    for email_item in data.get('value', []):
                        total_scanned_emails += 1

                        rcv_iso = email_item.get('receivedDateTime')
                        if not rcv_iso:
                            continue
                        # Normalize to aware UTC datetime, then enforce cutoff
                        rcv_dt = datetime.fromisoformat(rcv_iso.replace('Z', '+00:00')).astimezone(timezone.utc)
                        if rcv_dt < cutoff:
                            continue

                        # Keyword filter
                        # Build subject+body and filter on both
                        # ----- SUBJECT + BODY keyword check -----
                        email_content = (email_item.get('body') or {}).get('content', '') or ''
                        # strip HTML so we can match keywords reliably if body is HTML
                        body_text = re.sub(r'<[^>]+>', ' ', email_content)
                        subject = (email_item.get('subject') or '')
                        combined = f"{subject}\n{body_text}"

                        matches_email_text = self.has_rfp_keywords(combined)

                        # If subject/body didn't match, try attachment **filenames**
                        should_process = matches_email_text
                        if not should_process:
                            if self._attachments_match_keywords(user_email, email_item['id'], headers):
                                should_process = True

                        if not should_process:
                            continue  # skip this email entirely



                        # Existing attachments (from prior runs) for this email
                        existing_atts = list(existing_map.get((user_email, email_item['id']), []))
                        new_attachments = []

                        # Build SharePoint folder for this email
                        date_str = email_item['receivedDateTime'][:10]  # YYYY-MM-DD
                        subject_clean = self.clean_filename(email_item['subject'])
                        sp_folder = f"Emails/{user_email}/{date_str}_{subject_clean}"
                        spm = SharePointManager()

                        # ---- LINKS (upload only if NEW) ----
                        
                        extracted_links = self.extract_links_from_email(email_content)

                        seen_links_this_email = set()
                        for link in extracted_links:
                            norm_url = _norm_url(link['url'])
                            if not norm_url or norm_url in seen_links_this_email:
                                continue
                            seen_links_this_email.add(norm_url)

                            uniq = f"{email_item['id']}|link|{norm_url}"
                            if uniq in seen:
                                continue  # previously processed

                            link_res = spm.save_link_to_sharepoint(link['url'], link.get('title') or norm_url, sp_folder)
                            if link_res and link_res.get('status') == 'success':
                                new_attachments.append({
                                    'filename': link.get('title') or norm_url,
                                    'file_path': '',
                                    'file_size': 0,
                                    'url': link['url'],
                                    'domain': link.get('domain'),
                                    'type': 'link',
                                    'download_date': datetime.now(timezone.utc).isoformat(),
                                    'sharepoint_path': sp_folder,
                                    'sharepoint_web_url': (link_res or {}).get('web_url'),
                                    'sharepoint_download_url': (link_res or {}).get('download_url'),
                                })
                                total_new_attachments += 1
                                seen.add(uniq)

                        # ---- FILES (upload only if NEW) ----
                        if email_item.get('hasAttachments'):
                            att_url = f"https://graph.microsoft.com/v1.0/users/{user_email}/messages/{email_item['id']}/attachments"
                            att_resp = requests.get(att_url, headers=headers, timeout=10)
                            if att_resp.status_code == 200:
                                for a in att_resp.json().get('value', []):
                                    if a.get('@odata.type') != '#microsoft.graph.fileAttachment':
                                        continue
                                    att_id = a.get('id')
                                    uniq = f"{email_item['id']}|att|{att_id or a.get('name')}"
                                    if uniq in seen:
                                        continue

                                    try:
                                        blob = base64.b64decode(a['contentBytes'])
                                    except Exception:
                                        continue

                                    saved = self.save_attachment(blob, a['name'], date_str)
                                    up = spm.upload_file_to_sharepoint(blob, self.clean_filename(a['name']), sp_folder)
                                    if saved and up and up.get('status') == 'success':
                                        saved['type'] = 'file'
                                        saved['sharepoint_path'] = sp_folder
                                        saved['download_date'] = datetime.now(timezone.utc).isoformat()
                                        saved['sharepoint_web_url'] = (up or {}).get('web_url')
                                        saved['sharepoint_download_url'] = (up or {}).get('download_url')
                                        new_attachments.append(saved)
                                        total_new_attachments += 1
                                        seen.add(uniq)
                        
                        # Combine existing + new so UI shows everything
                        all_atts = _dedup_attachments(existing_atts + new_attachments)
                        attachment_names = []
                        for a in all_atts:
                            name = (a.get('filename') or a.get('file_path') or a.get('url') or '').strip()
                            if name:
                                attachment_names.append(name)
                        batch_all.append({
                            "email_id": email_item['id'],
                            "sender": email_item['sender']['emailAddress']['address'],
                            "subject": email_item['subject'],
                            "date": email_item['receivedDateTime'],
                            "account": user_email,
                            "attachments": all_atts,
                            "has_rfp_keywords": True,
                            "body_text": body_text[:4000],          # NEW: plain-text body preview (safe size)
                            "attachment_names": attachment_names     # NEW: flattened filenames/URLs

                        })

                    # Pagination
                    next_link = data.get('@odata.nextLink')
                    if next_link:
                        url = next_link
                        params = None  # nextLink already carries query
                    else:
                        break

            # Persist de-dupe state
            save_processed_index(processed)

            return {
                "status": "success",
                "message": (
                    f"Scanned {len(self.graph_config['user_emails'])} accounts • "
                    f"{total_scanned_emails} emails in last 3 months • "
                    f"{total_new_attachments} new attachments/links uploaded."
                ),
                "emails_found": len(batch_all),                 # ALL matching emails
                "attachments_downloaded": total_new_attachments, # only new uploads
                "fetched_emails": batch_all
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
from fastapi.responses import JSONResponse


@email_router.post("/fetch-emails")
async def fetch_emails(request: EmailFetchRequest = EmailFetchRequest()):

    """Fetch RFP/RFI/RFQ emails and download attachments (Graph only)."""
    try:
        print("📧 Starting email fetch process...")

        token = email_fetcher.get_graph_access_token()
        if not token:
            return {
                "status": "error",
                "message": "Microsoft Graph authentication failed. Check GRAPH_TENANT_ID, GRAPH_CLIENT_ID, GRAPH_CLIENT_SECRET and admin consent.",
                "emails_found": 0,
                "attachments_downloaded": 0,
                "fetched_emails": []
            }

        # Directly fetch via Graph (no auth pre-tests)
        result = email_fetcher.fetch_emails_graph() or {
            "status": "error",
            "message": "fetch_emails_graph returned no data",
            "emails_found": 0,
            "attachments_downloaded": 0,
            "fetched_emails": []
        }

        if result.get('status') == 'success':
            global real_fetched_emails
            real_fetched_emails = result['fetched_emails']
            save_real_emails_to_file(real_fetched_emails)

        return JSONResponse(content=result)


    except Exception as e:
        print(f"Error fetching emails: {e}")
        return {
            "status": "error",
            "message": str(e),
            "emails_found": 0,
            "attachments_downloaded": 0,
            "fetched_emails": []
        }


from datetime import datetime, timedelta, timezone

@email_router.get("/fetched-emails")
async def get_fetched_emails():
    """Get list of previously fetched emails (real-only, stable IDs)."""
    try:
        global real_fetched_emails
        if not real_fetched_emails:
            # No real data yet — return empty (no demo)
            return {"emails": [], "total_count": 0, "total_files": 0}

        emails_out = []
        total_files = 0

        for i, (acct, lst) in enumerate(_ordered_accounts(real_fetched_emails), 1):
            # sort account's emails by date desc
            lst_sorted = sorted(lst, key=lambda e: _parse_dt(e.get('date', '')), reverse=True)
            latest = lst_sorted[0] if lst_sorted else {}
            latest_date = latest.get('date', '')
            latest_subject = latest.get('subject', 'No Subject')
            files_count = sum(len(e.get('attachments') or []) for e in lst)

            emails_out.append({
                "id": i,                         # STABLE id based on sorted order
                "email": acct,
                "count": len(lst),
                "latest_subject": latest_subject,
                "latest_date": latest_date,
                "latest_files": files_count,
                "emails": lst_sorted,            # keep for debugging if needed
            })
            total_files += files_count

        return {
            "emails": emails_out,
            "total_count": len(real_fetched_emails),
            "total_files": total_files,
        }

    except Exception as e:
        print(f"Error getting fetched emails: {e}")
        return {"emails": [], "total_count": 0, "total_files": 0}



from datetime import datetime, timedelta, timezone

def _human_size(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return "0 Bytes"
    units = ["Bytes", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}" if units[i] != "Bytes" else f"{int(n)} Bytes"

@email_router.get("/email-attachments/{email_id}")
async def get_email_attachments(email_id: int):
    """Get attachments for a specific email account (stable IDs match fetched-emails)."""
    try:
        global real_fetched_emails
        if not real_fetched_emails:
            return {"email_id": email_id, "attachments": []}

        # Same ordering as /api/fetched-emails
        ordered = _ordered_accounts(real_fetched_emails)

        # Find the requested account by the 1..N card id
        target = None
        for i, (acct, lst) in enumerate(ordered, 1):
            if i == email_id:
                target = {"account": acct, "emails": lst}
                break
        if not target:
            return {"email_id": email_id, "attachments": []}

        # Collect ALL attachments from ALL emails of this account
        out_attachments = []
        for em in target["emails"]:
            email_subject = em.get("subject", "No Subject")
            email_sender  = em.get("sender", "Unknown Sender")
            email_date    = em.get("date", "")
            for att in em.get("attachments") or []:
                a = dict(att)  # shallow copy
                a.setdefault("email_subject", email_subject)
                a.setdefault("email_sender",  email_sender)
                a.setdefault("email_date",    email_date)
                out_attachments.append(a)

        # NOW de-dupe across the whole account and drop images
        deduped = []
        seen_keys = set()
        for a in out_attachments:
            if _is_image_attachment(a):
                continue
            k = _att_key(a)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            deduped.append(a)

        return {
            "email_id": email_id,
            "account": target["account"],
            "attachments": deduped,
        }

    except Exception as e:
        print(f"Error getting email attachments: {e}")
        return {"email_id": email_id, "attachments": []}


# at top if missing
from pathlib import Path
from fastapi import Query, HTTPException
from fastapi.responses import FileResponse
import mimetypes

@email_router.get("/attachment/view")
async def view_attachment(path: str = Query(..., description="File path returned in attachments")):
    base = Path("email_attachments").resolve()

    try:
        p = Path(path)
        requested = (p if p.is_absolute() else base / p).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    # Only serve files inside email_attachments
    if base != requested and base not in requested.parents:
        raise HTTPException(status_code=403, detail="Forbidden")

    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    mt, _ = mimetypes.guess_type(str(requested))
    mt = mt or "application/octet-stream"
    headers = {"Content-Disposition": f'inline; filename="{requested.name}"'}
    return FileResponse(str(requested), media_type=mt, headers=headers)
