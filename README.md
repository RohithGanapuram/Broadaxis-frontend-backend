# 🤖 BroadAxis RFP/RFQ Management Platform

A comprehensive AI-powered platform for analyzing, processing, and responding to RFPs (Request for Proposals), RFIs (Request for Information), and RFQs (Request for Quotations). Built with React TypeScript frontend, FastAPI Python backend, and MCP (Model Context Protocol) server for AI integration.

## 🏗️ Architecture Overview

```
React TypeScript Frontend (Port 5173)
    ↕ (REST API + WebSocket)
FastAPI Backend (Port 8000)
    ↕ (stdio communication)
MCP Server (server.py)
    ↕ (API calls)
AI Services (Claude, Pinecone, Tavily, OpenAI)
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+** with pip
- **Node.js 18+** with npm
- **Microsoft Graph API** credentials
- **AI Service API Keys** (Anthropic, Pinecone, Tavily, OpenAI)

### Installation & Setup

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd ba-rfpapp
   
   # Create Python virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

2. **Install Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   # AI Services
   ANTHROPIC_API_KEY=your_anthropic_key
   PINECONE_API_KEY=your_pinecone_key
   TAVILY_API_KEY=your_tavily_key
   OPENAI_API_KEY=your_openai_key
   
   # Microsoft Graph API
   GRAPH_CLIENT_ID=your_client_id
   GRAPH_CLIENT_SECRET=your_client_secret
   GRAPH_TENANT_ID=your_tenant_id
   
   # Email Accounts (up to 3)
   GRAPH_USER_EMAIL_1=email1@company.com
   GRAPH_USER_EMAIL_2=email2@company.com
   GRAPH_USER_EMAIL_3=email3@company.com
   
   # SharePoint Configuration
   SHAREPOINT_SITE_URL=company.sharepoint.com:/sites/project
   SHAREPOINT_FOLDER_PATH=Documents
   ```

### Running the Application

**Option 1: Run Services Separately**

```bash
# Terminal 1 - Backend API Server
cd backend
python run_backend.py
# Or: python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend Development Server
cd frontend
npm run dev
```

**Option 2: Access URLs**
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws/chat

## 🔧 Core Features

### 🤖 AI-Powered Analysis
- **Intelligent RFP Processing**: Upload documents and get instant AI analysis
- **Go/No-Go Recommendations**: AI-driven decision support with scoring
- **Company Knowledge Integration**: Leverages internal knowledge base via Pinecone
- **Multi-Model Support**: Claude 3.5 Sonnet, Claude 3 Haiku, and OpenAI models
- **Token Management**: Smart token counting with limits (8K/request, 50K/session, 200K/day)

### 📧 Email Integration & Management
- **Microsoft Graph API Integration**: Fetch RFP/RFI/RFQ emails from multiple accounts
- **Automatic Attachment Download**: PDF and document extraction with SharePoint storage
- **Email Authentication**: OAuth2 flow with proper permission handling
- **Real-time Email Processing**: Automatic detection of RFP-related emails
- **Attachment Management**: Organize and categorize email attachments

### 📄 Document Management & Processing
- **Multi-format Support**: PDF, DOCX, DOC, TXT, MD files
- **Real-time Processing**: Instant text extraction and analysis
- **SharePoint Integration**: Generated files automatically saved to SharePoint
- **Advanced PDF Processing**: Text extraction, metadata analysis, form field detection
- **File Operations**: Complete CRUD operations on SharePoint files

### 🔍 Research & Knowledge Management
- **Internal Knowledge Search**: Company expertise and past projects via Pinecone
- **Web Search Integration**: External market research via Tavily API
- **SharePoint Search**: Find files across SharePoint repositories
- **Document Embedding**: Automatic vectorization for semantic search

### ⚡ Performance & Reliability
- **Parallel Tool Execution**: Concurrent API calls using asyncio.gather
- **Connection Status Tracking**: Real-time MCP server status monitoring
- **Caching System**: SharePoint folder caching to reduce API calls
- **Error Recovery**: Comprehensive error handling with fallback mechanisms
- **WebSocket Communication**: Real-time chat and status updates

## 📁 Project Structure

```
ba-rfpapp/
├── .env                           # Environment variables
├── requirements.txt               # Python dependencies
├── embedding.py                   # PDF embedding utilities
├── README.md                      # This file
│
├── backend/                       # FastAPI Backend
│   ├── api.py                    # Main API endpoints (2500+ lines)
│   ├── error_handler.py          # Centralized error handling
│   ├── shared_utils.py           # Shared utilities
│   ├── run_backend.py            # Backend startup script
│   ├── real_fetched_emails.json  # Email data storage
│   └── email_attachments/        # Downloaded email attachments
│       ├── 2025-01-13/
│       ├── 2025-01-14/
│       └── ... (organized by date)
│
├── frontend/                      # React TypeScript Frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   │   ├── Auth/            # Authentication components
│   │   │   │   ├── Login.tsx
│   │   │   │   └── Register.tsx
│   │   │   ├── Dashboard/       # Dashboard components
│   │   │   │   ├── Dashboard.tsx
│   │   │   │   ├── StatsCards.tsx
│   │   │   │   └── RecentDocuments.tsx
│   │   │   ├── Email/           # Email management
│   │   │   │   └── Email.tsx
│   │   │   ├── SharedFolder/    # SharePoint file browser
│   │   │   │   └── SharedFolder.tsx
│   │   │   ├── Sidebar/         # Navigation sidebar
│   │   │   │   └── Sidebar.tsx
│   │   │   ├── DashboardLayout.tsx
│   │   │   └── Layout.tsx
│   │   ├── pages/               # Main page components
│   │   │   ├── ChatInterface.tsx # AI chat interface
│   │   │   ├── FileManager.tsx  # File management
│   │   │   └── Settings.tsx     # Application settings
│   │   ├── context/             # React context providers
│   │   │   ├── AppContext.tsx   # Global app state
│   │   │   └── AuthContext.tsx  # Authentication state
│   │   ├── utils/               # Utility functions
│   │   │   ├── api.ts           # API client
│   │   │   └── websocket.ts     # WebSocket client
│   │   ├── types/               # TypeScript definitions
│   │   │   └── index.ts
│   │   ├── App.tsx              # Main app component
│   │   ├── main.tsx             # App entry point
│   │   └── index.css            # Global styles
│   ├── package.json             # Frontend dependencies
│   ├── vite.config.ts           # Vite configuration
│   └── tailwind.config.js       # Tailwind CSS config
│
└── ba-server/                    # MCP Server
    ├── server.py                # Main MCP server (950+ lines)
    ├── generated_files/         # Generated documents
    └── mcp_server.log           # Server logs
```

## 🔌 API Endpoints

### Core Functionality
- `POST /api/initialize` - Initialize MCP server with tools & prompts
- `POST /api/chat` - Send chat messages with tool selection
- `POST /api/upload` - Upload files with validation and analysis
- `WS /ws/chat` - WebSocket for real-time chat communication

### Email & SharePoint Management
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

## 🛠️ Available MCP Tools

### Core AI Tools
1. **broadaxis_knowledge_search** - Internal company knowledge search via Pinecone
2. **web_search_tool** - External web search via Tavily API

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

## 🎯 Usage Workflows

### 1. RFP/RFQ Analysis Workflow
1. **Upload Document**: Drag & drop RFP document via chat interface
2. **Automatic Analysis**: Get instant AI analysis with key requirements
3. **Go/No-Go Decision**: Request AI recommendation with scoring
4. **Response Generation**: Generate professional response documents
5. **SharePoint Storage**: Files automatically saved to SharePoint

### 2. Email Management Workflow
1. **Fetch Emails**: Click "Fetch RFP/RFI/RFQ Emails" button
2. **Automatic Processing**: System searches configured email accounts
3. **Attachment Download**: Relevant attachments downloaded to SharePoint
4. **Organization**: Emails organized by date and account
5. **Analysis**: Process downloaded attachments for RFP analysis

### 3. Document Generation Workflow
1. **Template Selection**: Choose document type (PDF, Word, Text)
2. **Content Generation**: AI generates content based on RFP requirements
3. **Professional Formatting**: Documents formatted with company branding
4. **SharePoint Integration**: Files automatically saved to SharePoint
5. **Team Access**: Generated documents available for team collaboration

## 🔐 Authentication & Security

### Frontend Authentication
- **Local Storage**: User session management
- **Context Providers**: React context for auth state
- **Protected Routes**: Authentication-required components

### Backend Security
- **API Key Management**: Secure environment variable storage
- **CORS Configuration**: Proper frontend access control
- **File Upload Validation**: Type checking and sanitization
- **Rate Limiting**: Token-based usage limits and monitoring
- **Error Handling**: Comprehensive logging without sensitive data exposure

### Microsoft Graph Security
- **OAuth2 Authentication**: Proper authentication flow
- **Permission Scopes**: Minimal required permissions
- **Token Management**: Secure token handling and refresh

## 🚀 Deployment

### Production Setup
1. **Build Frontend**
   ```bash
   cd frontend
   npm run build
   ```

2. **Deploy Backend**
   ```bash
   # Install production dependencies
   pip install -r requirements.txt
   
   # Run with gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app
   ```

3. **Configure Reverse Proxy**
   ```nginx
   # Nginx configuration example
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:5173;
       }
       
       location /api {
           proxy_pass http://localhost:8000;
       }
   }
   ```

4. **Environment Configuration**
   - Set production environment variables
   - Configure Microsoft Graph API permissions
   - Set up SSL certificates

## 📊 Monitoring & Logging

### Health Monitoring
- **Health Check Endpoint**: `/health` - Detailed system status
- **Server Status**: `/api/status` - MCP server connection status
- **WebSocket Monitoring**: Real-time connection status
- **Token Usage Tracking**: `/api/tokens` - Usage statistics

### Logging
- **Backend Logs**: Console output and file logging
- **MCP Server Logs**: `ba-server/mcp_server.log`
- **Error Logs**: `backend/broadaxis_errors.log`
- **Frontend Logs**: Browser console and error tracking

## 🆘 Troubleshooting

### Common Issues & Solutions

1. **Email Fetching Fails**
   - **Issue**: Microsoft Graph API authentication errors
   - **Solution**: Check API credentials and permissions in Azure AD

2. **SharePoint Access Denied**
   - **Issue**: File operation permissions
   - **Solution**: Verify SharePoint site permissions and authentication

3. **MCP Server Offline**
   - **Issue**: Python dependencies or model loading
   - **Solution**: Check Python environment and sentence transformer model

4. **Frontend Build Errors**
   - **Issue**: Missing dependencies or TypeScript errors
   - **Solution**: Run `npm install` and check TypeScript compilation

5. **PDF Generation Hanging**
   - **Issue**: SharePoint connectivity or file permissions
   - **Solution**: Check SharePoint connectivity and fallback to local storage

6. **Slow Startup**
   - **Issue**: First run loads large AI models
   - **Solution**: Initial startup takes 60-90 seconds for model loading

7. **WebSocket Connection Issues**
   - **Issue**: Real-time communication failures
   - **Solution**: Check firewall settings and WebSocket endpoint

### Performance Optimization
- **Caching**: SharePoint folder caching reduces API calls
- **Parallel Processing**: Concurrent tool execution
- **Token Management**: Efficient token usage tracking
- **Connection Pooling**: Optimized HTTP connections

## 🔧 Development

### Backend Development
```bash
cd backend
python run_backend.py
# API available at http://localhost:8000
```

### Frontend Development
```bash
cd frontend
npm run dev
# Frontend available at http://localhost:5173
```

### MCP Server Testing
```bash
cd ba-server
python server.py
# Test MCP server directly
```

### Code Quality
- **TypeScript**: Strict type checking for frontend
- **Python**: Type hints and error handling
- **ESLint**: Code quality enforcement
- **Error Handling**: Comprehensive error management

## 📝 License

**Proprietary Software** - BroadAxis Internal Use Only

---

**Built with ❤️ for BroadAxis RFP/RFQ Management**

*Version 1.0.0 - Comprehensive AI-powered RFP management platform*