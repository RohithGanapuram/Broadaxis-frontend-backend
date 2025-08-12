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
from langchain_huggingface import HuggingFaceEmbeddings
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

#Connection to Pinecone
try:
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    if not pinecone_api_key:
        logger.warning("PINECONE_API_KEY not found in environment variables")
        pc = None
        index = None
    else:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("sample3")
        logger.info("Pinecone connection established")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    pc = None
    index = None

try:
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("HuggingFace embedder initialized")
except Exception as e:
    logger.error(f"Failed to initialize embedder: {e}")
    embedder = None

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
    Retrieves the most relevant company's(Broadaxus) information from the internal knowledge base in response to an company related query.

    This tool performs semantic search over a RAG-powered database containing details about the Broadaxis's background, team, projects, responsibilities, and domain expertise. It is designed to support tasks such as retreiving the knowledge regarding the company, surfacing domain-specific experience.

    Args:
        query: A natural language request related to the company’s past work, expertise, or capabilities (e.g., "What are the team's responsibilities?").
    """
    # try:
        # Step 1: Embed the query
    if not query or not query.strip():
        return json.dumps({"error": "Query cannot be empty"})
    
    if not embedder:
        return json.dumps({"error": "Embedder not available"})
    
    if not index:
        return json.dumps({"error": "Pinecone index not available"})
    
    try:
        query_embedding = embedder.embed_documents([query.strip()])[0]
        query_result = index.query(
            vector=[query_embedding],
            top_k=5,
            include_metadata=True,
            namespace=""
        )
        
        documents = [result['metadata']['text'] for result in query_result['matches'] if 'metadata' in result]
        return documents if documents else json.dumps({"message": "No relevant documents found"})
    except Exception as e:
        logger.error(f"Knowledge search error: {e}")
        return json.dumps({"error": f"Search failed: {str(e)}"})

    # except Exception as e:
    #     return json.dumps({"error": str(e)})

# Initialize client (use env variable or hardcode the API key if preferred)

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
def filesystem_read_file(path: str) -> str:
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
def filesystem_write_file(path: str, content: str) -> str:
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
def filesystem_list_directory(path: str = "") -> str:
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
def filesystem_delete_file(path: str) -> str:
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
def filesystem_search_files(query: str, path: str = "") -> str:
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
def pdffiller_extract_text(path: str) -> str:
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
def pdffiller_get_form_fields(path: str) -> str:
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



@mcp.prompt(title="Identifying the Documents")
def Step1_Identifying_documents():
    """read PDFs from filesystem path, categorize them as RFP/RFI/RFQ-related, fillable forms, or non-fillable documents."""
    return f"""read the files from the provided filesystems tool path using PDFFiller tool to categorize each uploaded PDF into the following groups:

1. 📘 **Primary Documents** — PDFs that contain RFP, RFQ, or RFI content (e.g., project scope, requirements, evaluation criteria).
2. 📝 **Fillable Forms** — PDFs with interactive fields intended for user input (e.g., pricing tables, response forms).
3. 📄 **Non-Fillable Documents** — PDFs that are neither RFP-type nor interactive, such as attachments or informational appendices.
---
Once the classification is complete:

📊 **Would you like to proceed to the next step and generate summaries for the relevant documents?**  
If yes, please upload the files and attach the summary prompt template.
"""


@mcp.prompt(title="Step-2: Executive Summary of Procurement Document")
def Step2_summarize_documents():
    """Generate a clear, high-value summary of uploaded RFP, RFQ, or RFI documents for executive decision-making."""
    return f"""
You are **BroadAxis-AI**, an intelligent assistant that analyzes procurement documents (RFP, RFQ, RFI) to help vendor teams quickly understand the opportunity and make informed pursuit decisions.
When a user uploads one or more documents, do the following **for each document, one at a time**:

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