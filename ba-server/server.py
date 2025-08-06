#server
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from mcp.server.fastmcp.prompts import base
from typing import TypedDict
import arxiv
import json
import os
from typing import List
from typing import Union
import re
import PyPDF2
from io import BytesIO
import mcp.types as types
import logging
import traceback
from datetime import datetime
try:
    import docx
except ImportError:
    docx = None
import tempfile
import base64
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp
from pydantic import BaseModel
from typing import Any, Sequence, List, Union
import httpx
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()
# File generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from fpdf import FPDF
import markdown
from jinja2 import Template
import uuid
import datetime
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


#constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

#Helper functions
def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    try:
        with httpx.Client() as client:
            response = client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
    except Exception:
        return None
        
def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""


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
PAPER_DIR = os.path.join(BASE_DIR, "papers")
GENERATED_FILES_DIR = os.path.join(BASE_DIR, "generated_files")

# Ensure directories exist
os.makedirs(PAPER_DIR, exist_ok=True)
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)


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
    Generate a PDF document with the provided title and content.

    Args:
        title: The title of the document
        content: The main content of the document (supports markdown formatting)
        filename: Optional custom filename (without extension)

    Returns:
        JSON string with file information including download path
    """
    try:
        # Generate unique filename if not provided
        if not filename:
            filename = f"document_{uuid.uuid4().hex[:8]}"

        # Ensure filename doesn't have extension
        filename = filename.replace('.pdf', '')

        # Create full file path
        file_path = os.path.join(GENERATED_FILES_DIR, f"{filename}.pdf")

        # Create PDF document
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))

        # Process content (convert markdown to HTML-like formatting for reportlab)
        content_lines = content.split('\n')
        for line in content_lines:
            if line.strip():
                # Handle basic markdown formatting
                if line.startswith('# '):
                    story.append(Paragraph(line[2:], styles['Heading1']))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], styles['Heading2']))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], styles['Heading3']))
                elif line.startswith('- ') or line.startswith('* '):
                    story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
                else:
                    story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))

        # Build PDF
        doc.build(story)

        # Get file info
        file_size = os.path.getsize(file_path)

        return json.dumps({
            "status": "success",
            "filename": f"{filename}.pdf",
            "file_path": file_path,
            "file_size": file_size,
            "download_url": f"/download/{filename}.pdf",
            "created_at": datetime.datetime.now().isoformat(),
            "type": "pdf"
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


@mcp.tool()
def generate_word_document(title: str, content: str, filename: str = None) -> str:
    """
    Generate a Word document with the provided title and content.

    Args:
        title: The title of the document
        content: The main content of the document (supports basic markdown formatting)
        filename: Optional custom filename (without extension)

    Returns:
        JSON string with file information including download path
    """
    try:
        # Generate unique filename if not provided
        if not filename:
            filename = f"document_{uuid.uuid4().hex[:8]}"

        # Ensure filename doesn't have extension
        filename = filename.replace('.docx', '').replace('.doc', '')

        # Create full file path
        file_path = os.path.join(GENERATED_FILES_DIR, f"{filename}.docx")

        # Create Word document
        doc = Document()

        # Add title
        title_paragraph = doc.add_heading(title, level=1)
        title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add some space after title
        doc.add_paragraph()

        # Process content (handle basic markdown formatting)
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
                    # Add bullet point
                    doc.add_paragraph(line[2:], style='List Bullet')
                elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. '):
                    # Add numbered list
                    doc.add_paragraph(line[3:], style='List Number')
                else:
                    # Regular paragraph
                    doc.add_paragraph(line)
            else:
                # Add empty paragraph for spacing
                doc.add_paragraph()

        # Save document
        doc.save(file_path)

        # Get file info
        file_size = os.path.getsize(file_path)

        return json.dumps({
            "status": "success",
            "filename": f"{filename}.docx",
            "file_path": file_path,
            "file_size": file_size,
            "download_url": f"/download/{filename}.docx",
            "created_at": datetime.datetime.now().isoformat(),
            "type": "docx"
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


@mcp.tool()
def generate_text_file(content: str, filename: str = None, file_extension: str = "txt") -> str:
    """
    Generate a text file with the provided content.

    Args:
        content: The content to write to the file
        filename: Optional custom filename (without extension)
        file_extension: File extension (txt, md, csv, json, etc.)

    Returns:
        JSON string with file information including download path
    """
    try:
        # Generate unique filename if not provided
        if not filename:
            filename = f"file_{uuid.uuid4().hex[:8]}"

        # Clean filename and ensure no extension
        filename = filename.split('.')[0]

        # Create full file path
        file_path = os.path.join(GENERATED_FILES_DIR, f"{filename}.{file_extension}")

        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Get file info
        file_size = os.path.getsize(file_path)

        return json.dumps({
            "status": "success",
            "filename": f"{filename}.{file_extension}",
            "file_path": file_path,
            "file_size": file_size,
            "download_url": f"/download/{filename}.{file_extension}",
            "created_at": datetime.datetime.now().isoformat(),
            "type": file_extension
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


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