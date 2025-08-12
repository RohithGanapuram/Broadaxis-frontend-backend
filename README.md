# 🤖 BroadAxis RFP/RFQ Management Platform

A modern AI-powered platform for analyzing, processing, and responding to RFPs and RFQs using React frontend, FastAPI backend, and MCP (Model Context Protocol) server.

## 🏗️ Architecture

```
React Frontend (Port 3000)
    ↕ (REST/WebSocket)
FastAPI Backend (Port 8000)
    ↕ (stdio)
MCP Server (server.py)
    ↕ (API calls)
AI Services (Claude, Pinecone, etc.)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- npm or yarn

### Installation & Setup

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Configure Environment Variables**
   - Ensure `.env` file exists in the root directory with your API keys
   - Required: `ANTHROPIC_API_KEY`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET`, `GRAPH_TENANT_ID`

### Running the Application

**Option 1: Run Services Separately**

```bash
# Terminal 1 - Backend
cd backend
python api.py

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

**Option 2: Access URLs**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🔧 Key Features

### 🤖 AI-Powered Analysis
- **Automatic RFP/RFQ Processing**: Upload documents and get instant analysis
- **Go/No-Go Recommendations**: AI-driven decision support with scoring
- **Company Knowledge Integration**: Leverages internal knowledge base via Pinecone
- **Token Management**: Smart token counting and usage limits (8K/request, 50K/session, 200K/day)

### 📧 Email Integration
- **Microsoft Graph API**: Fetch RFP/RFI/RFQ emails from multiple accounts
- **Automatic Attachment Download**: PDF and document extraction
- **SharePoint Integration**: Save attachments to SharePoint automatically
- **Email Authentication**: OAuth2 flow with proper permission handling

### 📄 Document Management
- **Multi-format Support**: PDF, DOCX, DOC, TXT, MD files
- **Real-time Processing**: Instant text extraction and analysis
- **SharePoint Storage**: Generated files automatically saved to SharePoint
- **PDF Processing**: Advanced PDF text extraction and form field detection
- **File Operations**: Complete CRUD operations on SharePoint files

### 🔍 Research Capabilities
- **Internal Knowledge Search**: Company expertise and past projects
- **Web Search Integration**: External market research via Tavily
- **SharePoint Search**: Find files across SharePoint repositories

### ⚡ Performance Features
- **Parallel Tool Execution**: Concurrent API calls using asyncio.gather
- **Connection Status Tracking**: Real-time MCP server status monitoring
- **Caching System**: SharePoint folder caching to reduce API calls
- **Error Recovery**: Comprehensive error handling with fallback mechanisms

## 📁 Project Structure

```
ba-rfpapp/
├── .env                    # Environment variables (root level)
├── .gitignore             # Git ignore rules
├── requirements.txt        # Consolidated Python dependencies
├── backend/               # FastAPI REST API
│   ├── api.py            # Main API endpoints with error handling
│   └── error_handler.py  # Centralized error handling system
├── frontend/             # React TypeScript App
│   ├── src/
│   │   ├── components/   # React components
│   │   │   ├── Email/    # Email management components
│   │   │   └── SharedFolder/ # SharePoint file browser
│   │   ├── pages/        # Page components
│   │   ├── utils/        # API client & utilities
│   │   └── types/        # TypeScript definitions
│   └── package.json
└── ba-server/            # MCP Server
    ├── server.py         # Main MCP server with all tools
    └── .gitignore        # Server-specific ignore rules
```

## 🔌 API Endpoints

### Core Functionality
- `POST /api/initialize` - Initialize MCP server (unified tools & prompts)
- `POST /api/chat` - Send chat messages with tool selection
- `POST /api/upload` - Upload files with validation
- `WS /ws/chat` - WebSocket for real-time chat

### Email & SharePoint
- `POST /api/fetch-emails` - Fetch RFP/RFI/RFQ emails via Graph API
- `GET /api/fetched-emails` - List fetched emails with metadata
- `GET /api/email-attachments/{id}` - Get email attachments
- `POST /api/test-auth` - Test Microsoft Graph authentication
- `GET /api/files` - List SharePoint files with caching
- `GET /api/files/{path}` - Browse SharePoint folders
- `GET /api/sharepoint-file/{path}` - Get SharePoint file content
- `DELETE /api/sharepoint-file/{path}` - Delete SharePoint files
- `POST /api/sharepoint-search` - Search SharePoint files

### System & Monitoring
- `GET /health` - Health check with detailed status
- `GET /api/status` - MCP server connection status
- `GET /api/tokens` - Token usage statistics and limits
- `GET /api/tools` - List available MCP tools
- `GET /api/prompts` - List available MCP prompts

## 🛠️ Available Tools

### Core AI Tools
1. **Broadaxis_knowledge_search** - Internal company knowledge search
2. **web_search_tool** - External web search via Tavily

### Document Generation Tools
3. **generate_pdf_document** - Professional PDF creation with ReportLab
4. **generate_word_document** - Word document generation with python-docx
5. **generate_text_file** - Text file creation with formatting

### SharePoint Filesystem Tools
6. **sharepoint_read_file** - Read file contents from SharePoint
7. **sharepoint_write_file** - Write/create files in SharePoint
8. **sharepoint_list_files** - List files and folders in SharePoint
9. **sharepoint_delete_file** - Delete files from SharePoint
10. **sharepoint_search_files** - Search files by name in SharePoint

### PDF Processing Tools
11. **extract_pdf_text** - Extract text content from PDF files
12. **get_pdf_metadata** - Get PDF document metadata
13. **detect_pdf_form_fields** - Detect form fields in PDF documents

## 🎯 Usage Workflow

### 1. RFP/RFQ Analysis
1. Upload RFP document via drag & drop
2. Get automatic analysis with key requirements
3. Request Go/No-Go recommendation
4. Generate response documents

### 2. Email Management
1. Click "Fetch RFP/RFI/RFQ Emails" button
2. System automatically searches configured email accounts
3. Downloads relevant attachments to SharePoint
4. View and manage fetched emails and attachments

### 3. Document Generation
1. Create professional PDFs and Word docs
2. Generate executive summaries
3. Build compliance matrices
4. Files automatically saved to SharePoint for team access

## 🔐 Environment Variables

Required variables in `.env` file:

```env
# AI Services
ANTHROPIC_API_KEY=your_anthropic_key
PINECONE_API_KEY=your_pinecone_key
TAVILY_API_KEY=your_tavily_key

# Microsoft Graph API
GRAPH_CLIENT_ID=your_client_id
GRAPH_CLIENT_SECRET=your_client_secret
GRAPH_TENANT_ID=your_tenant_id

# Email Accounts
GRAPH_USER_EMAIL_1=email1@company.com
GRAPH_USER_EMAIL_2=email2@company.com
GRAPH_USER_EMAIL_3=email3@company.com

# SharePoint
SHAREPOINT_SITE_URL=company.sharepoint.com:/sites/project
SHAREPOINT_FOLDER_PATH=Documents
```

## 🔧 Development

### Backend Development
```bash
cd backend
python api.py
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### MCP Server Testing
```bash
cd ba-server
python server.py
```

## 🚀 Deployment

### Production Setup
1. Build React frontend: `npm run build`
2. Deploy FastAPI with gunicorn
3. Configure reverse proxy (nginx)
4. Set up environment variables
5. Configure Microsoft Graph API permissions

## 📊 Monitoring

- Health check endpoint: `/health`
- Server status: `/api/status`
- WebSocket connection monitoring
- Token usage tracking: `/api/tokens`

## 🔒 Security Features

- **API Key Management**: Secure environment variable storage
- **CORS Configuration**: Proper frontend access control
- **File Upload Validation**: Type checking and sanitization
- **Rate Limiting**: Token-based usage limits and monitoring
- **Error Handling**: Comprehensive logging without sensitive data exposure
- **Microsoft Graph Security**: OAuth2 authentication with proper scopes
- **SharePoint Permissions**: Controlled file access and operations

## 🆘 Troubleshooting

### Common Issues
1. **Email fetching fails**: Check Microsoft Graph API credentials and permissions
2. **SharePoint access denied**: Verify site permissions and authentication
3. **MCP server offline**: Check Python dependencies and sentence transformer model
4. **Frontend build errors**: Run `npm install` in frontend directory
5. **PDF generation hanging**: Check SharePoint connectivity and fallback to local storage
6. **Unicode encoding errors**: Ensure proper character encoding in file operations
7. **Tool selection not persisting**: Clear browser localStorage if issues persist
8. **Slow startup**: First run loads 90MB+ sentence transformer model (60-90 seconds)

### Logs
- Backend logs: Console output
- MCP server logs: `ba-server/mcp_server.log`
- Frontend logs: Browser console

## 📝 License

Proprietary - BroadAxis Internal Use Only

---

**Built with ❤️ for BroadAxis RFP/RFQ Management**