#server
from mcp.server.fastmcp import FastMCP
import json
import os
import sys
from typing import List
import re
import logging
import traceback
from datetime import datetime
import tempfile

from pinecone import Pinecone
from tavily import TavilyClient
from dotenv import load_dotenv
# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
# File generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import uuid
import datetime
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
import PyPDF2
from io import BytesIO
from openai import OpenAI



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MCPServer')

# Error handling decorator
def handle_tool_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return json.dumps({
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
    return wrapper


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"  # 1536-dim
#Connection to Pinecone
try:
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    if not pinecone_api_key:
        logger.warning("PINECONE_API_KEY not found in environment variables")
        pc = None
        index = None
    else:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("final-index")
        
        # Test the connection and check index stats
        try:
            stats = index.describe_index_stats()
            logger.info(f"Pinecone connection established. Index stats: {stats}")
            
            # Check if broadaxis-index namespace exists
            namespaces = stats.get('namespaces', {})
            if 'broadaxis-index' in namespaces:
                vector_count = namespaces['broadaxis-index'].get('vector_count', 0)
                logger.info(f"Found {vector_count} vectors in 'broadaxis-index' namespace")
            else:
                logger.warning("'broadaxis-index' namespace not found. Available namespaces: " + str(list(namespaces.keys())))
                
        except Exception as stats_error:
            logger.warning(f"Could not get index stats: {stats_error}")
            
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    pc = None
    index = None

def _embed_text(text: str):
    """Return a single OpenAI embedding vector for the given text."""
    try:
        # Ensure text is not empty and clean it
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        clean_text = text.strip()
        logger.info(f"Embedding text of length: {len(clean_text)}")
        
        # Enhance query for better semantic matching
        enhanced_query = clean_text
        if any(word in clean_text.lower() for word in ["job", "duties", "responsibilities", "role"]):
            enhanced_query = f"Job responsibilities and duties: {clean_text}"
        elif any(word in clean_text.lower() for word in ["service", "capability", "expertise", "what does"]):
            enhanced_query = f"Company services and capabilities: {clean_text}"
        
        # Create high-quality embedding
        resp = client.embeddings.create(
            model=EMBED_MODEL, 
            input=[enhanced_query]
            # dimensions parameter not needed - uses model default (3072)
        )
        embedding = resp.data[0].embedding
        
        logger.info(f"Generated embedding with dimension: {len(embedding)}")
        return embedding  # list[float], length 1536
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

def sanitize_filename(name: str) -> str:
    # Remove all non-alphanumeric, dash, underscore characters
    return re.sub(r'[^a-zA-Z0-9-_]', '_', name)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


#create an MCP server
mcp = FastMCP("Research-Demo")

#Add an addition tool
@mcp.tool()
def sum(a: int, b: int) -> int:
    """Add two numbers"""
    return a+b


@mcp.tool()
def Broadaxis_knowledge_search(query: str):
    """
    Retrieves the most relevant company's (Broadaxis) information from the internal knowledge base.

    Semantic search over Pinecone using OpenAI embeddings.
    """
    if not query or not query.strip():
        return json.dumps({"error": "Query cannot be empty"})

    # Make sure we have an index
    if not index:
        return json.dumps({"error": "Pinecone index not available"})

    try:
        logger.info(f"Searching for: {query.strip()}")
        
        # Step 1: Embed the query with OpenAI
        query_embedding = _embed_text(query.strip())
        logger.info(f"Query embedding dimension: {len(query_embedding)}")

        # Step 2: Query Pinecone with optimized parameters
        query_result = index.query(
            vector=query_embedding,
            top_k=3,  # Fewer, higher quality results
            include_metadata=True,
            namespace="broadaxis-index",
            filter={}  # Can add filters if needed
        )
        
        logger.info(f"Pinecone query result type: {type(query_result)}")
        logger.info(f"Raw query result: {query_result}")

        # Step 3: Extract matches properly
        matches = []
        if hasattr(query_result, 'matches'):
            matches = query_result.matches
        elif isinstance(query_result, dict) and 'matches' in query_result:
            matches = query_result['matches']
        
        logger.info(f"Found {len(matches)} matches")

        # Step 4: Extract and return clean text content for LLM
        documents = []
        for i, match in enumerate(matches):
            try:
                # Get metadata
                metadata = {}
                if hasattr(match, 'metadata'):
                    metadata = match.metadata or {}
                elif isinstance(match, dict) and 'metadata' in match:
                    metadata = match['metadata'] or {}
                
                # Get score for logging
                score = 0.0
                if hasattr(match, 'score'):
                    score = match.score
                elif isinstance(match, dict) and 'score' in match:
                    score = match['score']
                
                # Get text content
                text_content = metadata.get('text', '')
                source = metadata.get('source', 'Unknown')
                
                if text_content:
                    # Clean text for LLM consumption
                    clean_text = text_content.replace('\uf0b7', '•').replace('\u2022', '•')
                    documents.append(clean_text)
                    logger.info(f"Match {i+1}: score={score:.4f}, source={source}, text_len={len(text_content)}")
                
            except Exception as match_error:
                logger.error(f"Error processing match {i}: {match_error}")
                continue

        if documents:
            # Return clean text content separated by double newlines
            return "\n\n".join(documents)
        else:
            return "No relevant information found in the BroadAxis knowledge base."

    except Exception as e:
        logger.error(f"Knowledge search error: {e}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        return f"Error searching BroadAxis knowledge base: {str(e)}"

os.environ["TAVILY_API_KEY"] = "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"
tavily = TavilyClient()  # or TavilyClient(api_key="your_key")

@mcp.tool()
def web_search_tool(query: str):
    """
    Performs a real-time web search using Tavily and returns relevant results
    (including title, URL, and snippet).

    Args:
        query: A natural language search query.

    Returns:
        A JSON string with the top search results.
    """
    try:
        if not query or not query.strip():
            return json.dumps({"error": "Search query cannot be empty"})
        
        results = tavily.search(query=query.strip(), search_depth="advanced", include_answer=False)
        formatted = [
            {
                "title": r.get("title", "No title"),
                "url": r.get("url", ""),
                "snippet": r.get("content", "No content")
            }
            for r in results.get("results", [])
        ]

        return json.dumps({"results": formatted, "total_results": len(formatted)})

    except Exception as e:
        logger.error(f"Web search error: {e}")
        return json.dumps({"error": f"Search failed: {str(e)}"})
    
@mcp.tool()
def generate_pdf_document(title: str, content: str, filename: str = None) -> str:
    """
    Generate and create a PDF document from text content. Use this tool when the user asks to create, generate, or make a PDF file.

    Args:
        title: The title of the document
        content: The main content of the document (supports markdown formatting)
        filename: Optional custom filename (without extension)

    Returns:
        JSON string with file generation status
    """
    logger.info(f"Starting PDF generation: {title}")
    temp_file_path = None
    try:
        # Generate unique filename if not provided
        if not filename:
            filename = f"document_{uuid.uuid4().hex[:8]}"

        # Ensure filename doesn't have extension
        filename = sanitize_filename(filename.replace('.pdf', ''))

        # Create output directory if it doesn't exist
        output_dir = os.path.join(BASE_DIR, "generated_files")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create file path
        temp_file_path = os.path.join(output_dir, f"{filename}.pdf")
        
        # Create PDF document
        doc = SimpleDocTemplate(temp_file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))

        # Process content - simplified
        content_lines = content.split('\n')
        for line in content_lines:
            line = line.strip()
            if line:
                # Escape special characters for ReportLab
                line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                if line.startswith('# '):
                    story.append(Paragraph(line[2:], styles['Heading1']))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], styles['Heading2']))
                elif line.startswith('- ') or line.startswith('* '):
                    story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
                else:
                    story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))

        # Build PDF
        doc.build(story)
        
        # Get file size
        file_size = os.path.getsize(temp_file_path)
        
        # Upload to SharePoint
        try:
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
            from api import SharePointManager  # type: ignore
            
            with open(temp_file_path, 'rb') as f:
                file_content = f.read()
            
            sharepoint_manager = SharePointManager()
            sharepoint_folder = f"Generated_Documents/{datetime.datetime.now().strftime('%Y-%m')}"
            
            logger.info(f"Attempting SharePoint upload: {filename}.pdf to {sharepoint_folder}")
            upload_result = sharepoint_manager.upload_file_to_sharepoint(
                file_content, f"{filename}.pdf", sharepoint_folder
            )
            logger.info(f"SharePoint upload result: {upload_result}")
            
            if upload_result['status'] == 'success':
                os.unlink(temp_file_path)  # Only delete if SharePoint upload succeeds
                return json.dumps({
                    "status": "success",
                    "filename": f"{filename}.pdf",
                    "sharepoint_path": sharepoint_folder,
                    "file_size": file_size,
                    "created_at": datetime.datetime.now().isoformat(),
                    "type": "pdf",
                    "message": "PDF generated and saved to SharePoint"
                })
            else:
                logger.warning(f"SharePoint upload failed: {upload_result.get('message', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"SharePoint upload failed: {e}")
        
        # Return success with local file path (SharePoint failed)
        return json.dumps({
            "status": "success",
            "filename": f"{filename}.pdf",
            "file_path": temp_file_path,
            "file_size": file_size,
            "created_at": datetime.datetime.now().isoformat(),
            "type": "pdf",
            "message": "PDF generated locally (SharePoint unavailable)"
        })

    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        # Clean up on error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return json.dumps({
            "status": "error",
            "error": str(e),
            "message": "Failed to generate PDF document"
        })


@mcp.tool()
def generate_word_document(title: str, content: str, filename: str = None) -> str:
    """
    Generate a Word document and save it to SharePoint.

    Args:
        title: The title of the document
        content: The main content of the document (supports basic markdown formatting)
        filename: Optional custom filename (without extension)

    Returns:
        JSON string with SharePoint upload status and file information
    """
    try:
        # Generate unique filename if not provided
        if not filename:
            filename = f"document_{uuid.uuid4().hex[:8]}"

        # Ensure filename doesn't have extension
        filename = filename.replace('.docx', '').replace('.doc', '')

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
        
        # Create Word document
        doc = Document()

        # Add title
        title_paragraph = doc.add_heading(title, level=1)
        title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add some space after title
        doc.add_paragraph()

        # Process content
        content_lines = content.split('\n')
        for line in content_lines:
            line = line.strip()
            if line:
                if line.startswith('# '):
                    doc.add_heading(line[2:], level=1)
                elif line.startswith('## '):
                    doc.add_heading(line[3:], level=2)
                elif line.startswith('### '):
                    doc.add_heading(line[4:], level=3)
                elif line.startswith('- ') or line.startswith('* '):
                    doc.add_paragraph(line[2:], style='List Bullet')
                elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. '):
                    doc.add_paragraph(line[3:], style='List Number')
                else:
                    doc.add_paragraph(line)
            else:
                doc.add_paragraph()

        # Save to temp file
        doc.save(temp_file.name)
        
        # Read file content for SharePoint upload
        with open(temp_file.name, 'rb') as f:
            file_content = f.read()
        
        os.unlink(temp_file.name)
        
        # Upload to SharePoint
        try:
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
            from api import SharePointManager  # type: ignore
            
            sharepoint_manager = SharePointManager()
            sharepoint_folder = f"Generated_Documents/{datetime.datetime.now().strftime('%Y-%m')}"
            
            upload_result = sharepoint_manager.upload_file_to_sharepoint(
                file_content, f"{filename}.docx", sharepoint_folder
            )
            
            if upload_result['status'] == 'success':
                return json.dumps({
                    "status": "success",
                    "filename": f"{filename}.docx",
                    "sharepoint_path": sharepoint_folder,
                    "file_size": len(file_content),
                    "created_at": datetime.datetime.now().isoformat(),
                    "type": "docx",
                    "message": "Word document generated and saved to SharePoint"
                })
        except Exception as e:
            logger.warning(f"SharePoint upload failed: {e}")
        
        return json.dumps({
            "status": "success",
            "filename": f"{filename}.docx",
            "file_size": len(file_content),
            "created_at": datetime.datetime.now().isoformat(),
            "type": "docx",
            "message": "Word document generated locally (SharePoint unavailable)"
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


@mcp.tool()
def sharepoint_read_file(path: str) -> str:
    """
    Read a file from SharePoint.
    
    Args:
        path: SharePoint file path (e.g., "Documents/file.txt")
    
    Returns:
        JSON string with file content or error
    """
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
        from api import SharePointManager  # type: ignore
        
        sharepoint_manager = SharePointManager()
        result = sharepoint_manager.get_file_content(path)
        
        if result['status'] == 'success':
            return json.dumps({
                "status": "success",
                "content": result['content'],
                "path": path,
                "size": len(result['content'])
            })
        else:
            return json.dumps({"status": "error", "error": result['message']})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def sharepoint_write_file(path: str, content: str) -> str:
    """
    Write/create a file in SharePoint.
    
    Args:
        path: SharePoint file path (e.g., "Documents/file.txt")
        content: File content to write
    
    Returns:
        JSON string with operation status
    """
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
        from api import SharePointManager  # type: ignore
        
        sharepoint_manager = SharePointManager()
        folder_path = '/'.join(path.split('/')[:-1]) if '/' in path else ''
        filename = path.split('/')[-1]
        
        result = sharepoint_manager.upload_file_to_sharepoint(
            content.encode('utf-8'), filename, folder_path
        )
        
        if result['status'] == 'success':
            return json.dumps({
                "status": "success",
                "path": path,
                "size": len(content.encode('utf-8')),
                "message": "File written successfully"
            })
        else:
            return json.dumps({"status": "error", "error": result['message']})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def sharepoint_list_files(path: str = "") -> str:
    """
    List files and directories in SharePoint folder.
    
    Args:
        path: SharePoint folder path (empty for root)
    
    Returns:
        JSON string with directory listing
    """
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
        from api import SharePointManager  # type: ignore
        
        sharepoint_manager = SharePointManager()
        result = sharepoint_manager.list_files(path)
        
        if result['status'] == 'success':
            return json.dumps({
                "status": "success",
                "path": path,
                "items": result['files']
            })
        else:
            return json.dumps({"status": "error", "error": result['message']})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def sharepoint_delete_file(path: str) -> str:
    """
    Delete a file from SharePoint.
    
    Args:
        path: SharePoint file path to delete
    
    Returns:
        JSON string with operation status
    """
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
        from api import SharePointManager  # type: ignore
        
        sharepoint_manager = SharePointManager()
        result = sharepoint_manager.delete_file(path)
        
        if result['status'] == 'success':
            return json.dumps({
                "status": "success",
                "path": path,
                "message": "File deleted successfully"
            })
        else:
            return json.dumps({"status": "error", "error": result['message']})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def sharepoint_search_files(query: str, path: str = "") -> str:
    """
    Search for files in SharePoint by name.
    
    Args:
        query: Search query (filename pattern)
        path: SharePoint folder path to search in
    
    Returns:
        JSON string with search results
    """
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
        from api import SharePointManager  # type: ignore
        
        sharepoint_manager = SharePointManager()
        result = sharepoint_manager.search_files(query, path)
        
        if result['status'] == 'success':
            return json.dumps({
                "status": "success",
                "query": query,
                "path": path,
                "results": result['files']
            })
        else:
            return json.dumps({"status": "error", "error": result['message']})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def extract_pdf_text(path: str) -> str:
    """
    Extract text content from a PDF file in SharePoint.
    
    Args:
        path: SharePoint path to PDF file
    
    Returns:
        JSON string with extracted text
    """
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
        from api import SharePointManager  # type: ignore
        import PyPDF2
        from io import BytesIO
        
        sharepoint_manager = SharePointManager()
        file_result = sharepoint_manager.get_file_content(path, binary=True)
        
        if file_result['status'] != 'success':
            return json.dumps({"status": "error", "error": file_result['message']})
        
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_result['content']))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return json.dumps({
            "status": "success",
            "path": path,
            "text": text.strip(),
            "pages": len(pdf_reader.pages)
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def detect_pdf_form_fields(path: str) -> str:
    """
    Get fillable form fields from a PDF file in SharePoint.
    
    Args:
        path: SharePoint path to PDF file
    
    Returns:
        JSON string with form fields information
    """
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
        from api import SharePointManager  # type: ignore
        import PyPDF2
        from io import BytesIO
        
        sharepoint_manager = SharePointManager()
        file_result = sharepoint_manager.get_file_content(path, binary=True)
        
        if file_result['status'] != 'success':
            return json.dumps({"status": "error", "error": file_result['message']})
        
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_result['content']))
        fields = []
        
        if '/AcroForm' in pdf_reader.trailer['/Root']:
            form = pdf_reader.trailer['/Root']['/AcroForm']
            if '/Fields' in form:
                for field in form['/Fields']:
                    field_obj = field.get_object()
                    field_name = field_obj.get('/T', 'Unknown')
                    field_type = field_obj.get('/FT', 'Unknown')
                    fields.append({
                        "name": str(field_name),
                        "type": str(field_type)
                    })
        
        return json.dumps({
            "status": "success",
            "path": path,
            "has_form_fields": len(fields) > 0,
            "fields": fields
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def get_pdf_metadata(path: str) -> str:
    """
    Get metadata from a PDF file in SharePoint.
    
    Args:
        path: SharePoint path to PDF file
    
    Returns:
        JSON string with PDF metadata
    """
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
        from api import SharePointManager  # type: ignore
        import PyPDF2
        from io import BytesIO
        
        sharepoint_manager = SharePointManager()
        file_result = sharepoint_manager.get_file_content(path, binary=True)
        
        if file_result['status'] != 'success':
            return json.dumps({"status": "error", "error": file_result['message']})
        
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_result['content']))
        metadata = pdf_reader.metadata
        
        return json.dumps({
            "status": "success",
            "path": path,
            "pages": len(pdf_reader.pages),
            "title": str(metadata.get('/Title', 'Unknown')) if metadata else 'Unknown',
            "author": str(metadata.get('/Author', 'Unknown')) if metadata else 'Unknown',
            "subject": str(metadata.get('/Subject', 'Unknown')) if metadata else 'Unknown',
            "creator": str(metadata.get('/Creator', 'Unknown')) if metadata else 'Unknown',
            "producer": str(metadata.get('/Producer', 'Unknown')) if metadata else 'Unknown'
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def process_uploaded_document(file_content: str, filename: str, session_id: str) -> str:
    """
    Process uploaded document by extracting text, creating chunks, and storing in Pinecone.
    
    Args:
        file_content: Base64 encoded file content
        filename: Name of the uploaded file
        session_id: Session identifier for namespace isolation
    
    Returns:
        JSON string with processing status
    """
    try:
        import base64
        
        # Decode file content
        file_bytes = base64.b64decode(file_content)
        
        # Extract text based on file type
        file_ext = os.path.splitext(filename.lower())[1]
        
        if file_ext == '.pdf':
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif file_ext == '.txt':
            text = file_bytes.decode('utf-8', errors='ignore')
        elif file_ext == '.md':
            text = file_bytes.decode('utf-8', errors='ignore')
        elif file_ext in ['.docx']:
            from docx import Document
            doc = Document(BytesIO(file_bytes))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            text = file_bytes.decode('utf-8', errors='ignore')
        
        if not text.strip():
            return json.dumps({"status": "error", "error": "No text content found in document"})
        
        # Create chunks (1200 chars with 200 overlap)
        chunks = []
        chunk_size = 1200
        overlap = 200
        
        if len(text) <= chunk_size:
            chunks.append(text.strip())
        else:
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
        
        # Create embeddings
        embeddings = []
        for chunk in chunks:
            embedding = _embed_text(chunk)
            embeddings.append(embedding)
        
        # Store in Pinecone using session-based namespace
        if index and embeddings:
            namespace = f"session_{session_id}"
            vectors = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{filename}_{i}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "filename": filename,
                        "chunk_index": i,
                        "session_id": session_id
                    }
                })
            
            # Upsert vectors
            index.upsert(vectors=vectors, namespace=namespace)
            
            return json.dumps({
                "status": "success",
                "filename": filename,
                "chunks_created": len(chunks),
                "namespace": namespace,
                "message": f"Document processed and stored with {len(chunks)} chunks"
            })
        else:
            return json.dumps({"status": "error", "error": "Pinecone index not available"})
            
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def query_uploaded_documents(query: str, session_id: str) -> str:
    """
    Query uploaded documents using vector similarity search.
    
    Args:
        query: Search query
        session_id: Session identifier to search within
    
    Returns:
        JSON string with relevant document chunks
    """
    try:
        if not query.strip():
            return json.dumps({"status": "error", "error": "Query cannot be empty"})
        
        if not index:
            return json.dumps({"status": "error", "error": "Pinecone index not available"})
        
        # Create query embedding
        query_embedding = _embed_text(query.strip())
        
        # Search in session namespace
        namespace = f"session_{session_id}"
        
        search_result = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace=namespace
        )
        
        # Extract relevant chunks
        relevant_chunks = []
        for match in search_result.matches:
            if match.score > 0.5:  # Only include relevant matches
                relevant_chunks.append({
                    "text": match.metadata["text"],
                    "filename": match.metadata["filename"],
                    "score": match.score,
                    "chunk_index": match.metadata["chunk_index"]
                })
        
        if relevant_chunks:
            # Return formatted context for AI processing
            context = "\n\n".join([f"From {chunk['filename']}:\n{chunk['text']}" for chunk in relevant_chunks])
            return f"Based on your uploaded documents:\n\n{context}\n\nQuery: {query}"
        else:
            return "No relevant information found in your uploaded documents for this query."
            
    except Exception as e:
        logger.error(f"Document query error: {e}")
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def cleanup_session_vectors(session_id: str) -> str:
    """
    Clean up all vectors for a specific session from Pinecone.
    
    Args:
        session_id: Session identifier to clean up
    
    Returns:
        JSON string with cleanup status
    """
    try:
        if not index:
            return json.dumps({"status": "error", "error": "Pinecone index not available"})
        
        namespace = f"session_{session_id}"
        
        # Delete all vectors in the session namespace
        index.delete(delete_all=True, namespace=namespace)
        
        return json.dumps({
            "status": "success",
            "session_id": session_id,
            "namespace": namespace,
            "message": "Session vectors cleaned up successfully"
        })
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return json.dumps({"status": "error", "error": str(e)})


@mcp.prompt(title="Identifying the Documents")
def Step1_Identifying_documents():
    """Browse SharePoint folders to identify and categorize RFP/RFI/RFQ documents from available folders."""
    return f"""I'll help you identify and categorize documents from your SharePoint folders. Let me start by browsing the available folders to find RFP/RFQ/RFI documents.

**Step 1: Browse SharePoint Root Directory**
First, let me check what folders are available in your SharePoint:

*[I will use sharepoint_list_files tool to show available folders]*

**Step 2: Select Document Category**
Once I see the available folders (like RFP, RFQ, RFI, etc.), please tell me which folder you'd like to work with.

**Step 3: Browse Selected Folder**
I'll then list the subfolders/projects within your chosen category.

**Step 4: Analyze Documents**
After you select a specific project folder, I'll read and categorize each PDF file into:

1. 📘 **Primary Documents** — PDFs containing RFP, RFQ, or RFI content (project scope, requirements, evaluation criteria)
2. 📝 **Fillable Forms** — PDFs with interactive fields for user input (pricing tables, response forms)
3. 📄 **Non-Fillable Documents** — Supporting documents, attachments, or informational appendices

Let me start by checking your SharePoint folder structure...
"""


@mcp.prompt(title="Step-2: Executive Summary of Procurement Document")
def Step2_summarize_documents():
    """Generate a clear, high-value summary of uploaded RFP, RFQ, or RFI documents for executive decision-making."""
    return f"""
You are **BroadAxis-AI**, an intelligent assistant that analyzes procurement documents (RFP, RFQ, RFI) to help vendor teams quickly understand the opportunity and make informed pursuit decisions.
When analyzing documents from SharePoint folders, do the following **for each document, one at a time**:

---

### 📄 Document: [Document Name]

#### 🔹 What is This About?
> A 3–5 sentence **plain-English overview** of the opportunity. Include:
- Who issued it (organization)
- What they need / are requesting
- Why (the business problem or goal)
- Type of response expected (proposal, quote, info)

---

#### 🧩 Key Opportunity Details
List all of the following **if available** in the document:
- **Submission Deadline:** [Date + Time]
- **Project Start/End Dates:** [Dates or Duration]
- **Estimated Value / Budget:** [If stated]
- **Response Format:** (e.g., PDF proposal, online portal, pricing form, etc.)
- **Delivery Location(s):** [City, Region, Remote, etc.]
- **Eligibility Requirements:** (Certifications, licenses, location limits)
- **Scope Summary:** (Bullet points or short paragraph outlining main tasks or deliverables)

---

#### 📊 Evaluation Criteria
How will responses be scored or selected? Include weighting if provided (e.g., 40% price, 30% experience).

---

#### ⚠️ Notable Risks or Challenges
Mention anything that could pose a red flag or require clarification (tight timeline, vague scope, legal constraints, strict eligibility).

---

#### 💡 Potential Opportunities or Differentiators
Highlight anything that could give a competitive edge or present upsell/cross-sell opportunities (e.g., optional services, innovation clauses, incumbent fatigue).

---

#### 📞 Contact & Submission Info
- **Primary Contact:** Name, title, email, phone (if listed)
- **Submission Instructions:** Portal, email, physical, etc.

---

### 🤔 Ready for Action?
> Would you like a strategic assessment or a **Go/No-Go recommendation** for this opportunity?

⚠️ Only summarize what is clearly and explicitly stated. Never guess or infer.
"""


@mcp.prompt(title="Step-3 : Go/No-Go Recommendation")
def Step3_go_no_go_recommendation() -> str:
    return """
You are BroadAxis-AI, an assistant trained to evaluate whether BroadAxis should pursue an RFP, RFQ, or RFI opportunity.
The user has uploaded one or more opportunity documents. You have already summarized them/if not ask for the user to upload RFP/RFI/RF documents and generate summary.
Now perform a structured **Go/No-Go analysis** using the following steps:
---
### 🧠 Step-by-Step Evaluation Framework

1. **Review the RFP Requirements**
   - Highlight the most critical needs and evaluation criteria.

2. **Search Internal Knowledge** (via Broadaxis_knowledge_search)
   - Identify relevant past projects
   - Retrieve proof of experience in similar domains
   - Surface known strengths or capability gaps

3. **Evaluate Capability Alignment**
   - Estimate percentage match (e.g., "BroadAxis meets ~85% of the requirements")
   - Note any missing capabilities or unclear requirements

4. **Assess Resource Requirements**
   - Are there any specialized skills, timelines, or staffing needs?
   - Does BroadAxis have the necessary team or partners?

5. **Evaluate Competitive Positioning**
   - Based on known experience and domain, would BroadAxis be competitive?

Use only verified internal information (via Broadaxis_knowledge_search) and the uploaded documents.
Do not guess or hallucinate capabilities. If information is missing, clearly state what else is needed for a confident decision.
if your recommendation is a Go, list down the things to the user of the tasks he need to complete  to finish the submission of RFP/RFI/RFQ. 

"""

@mcp.prompt(title="Step-4 : Generate Proposal or Capability Statement")
def Step4_generate_capability_statement() -> str:
    return """
You are BroadAxis-AI, an assistant trained to generate high-quality capability statements and proposal documents for RFP and RFQ responses.
The user has either uploaded an opportunity document or requested a formal proposal. Use all available information from:

- Uploaded documents (RFP/RFQ)
- Internal knowledge (via Broadaxis_knowledge_search)
- Prior summaries or analyses already provided

---

### 🧠 Instructions

- Do not invent names, projects, or facts.
- Use Broadaxis_knowledge_search to populate all relevant content.
- Leave placeholders where personal or business info is not available.
- Maintain professional, confident, and compliant tone.

If this proposal is meant to be saved, offer to generate a PDF or Word version using the appropriate tool.

"""

@mcp.prompt(title="Step-5 : Fill in Missing Information")
def Step5_fill_missing_information() -> str:
    return """
You are BroadAxis-AI, an intelligent assistant designed to fill in missing fields using ppdf filler tool , answer RFP/RFQ questions, and complete response templates **strictly using verified information**.
 Your task is to **complete the missing sections** on the fillable documents which you have identified previously with reliable information from:

1. Broadaxis_knowledge_search (internal database)
2. The uploaded document itself
3. Prior chat context (if available)
---
### 🧠 RULES (Strict Compliance)

- ❌ **DO NOT invent or hallucinate** company details, financials, certifications, team names, or client info.
- ❌ **DO NOT guess** values you cannot verify.
- 🔐 If the question involves **personal, legal, or confidential information**, **do not fill it**.
- ✅ Use internal knowledge only when it clearly answers the field.
---
### ✅ Final Instruction
Only fill what you can verify using Broadaxis_knowledge_search and uploaded content. Leave everything else with clear, professional placeholders.

"""


if __name__ == "__main__":
    import sys
    import logging

    # Set up logging for debugging
    logging.basicConfig(level=logging.INFO)

    try:
        # Run the MCP server synchronously
        mcp.run()
    except KeyboardInterrupt:
        print("Server stopped", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)