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
- **Go/No-Go Recommendations**: AI-driven decision support
- **Company Knowledge Integration**: Leverages internal knowledge base via Pinecone

### 📧 Email Integration
- **Microsoft Graph API**: Fetch RFP/RFI/RFQ emails from multiple accounts
- **Automatic Attachment Download**: PDF and document extraction
- **SharePoint Integration**: Save attachments to SharePoint automatically

### 📄 Document Management
- **Multi-format Support**: PDF, DOCX, DOC, TXT, MD files
- **Real-time Processing**: Instant text extraction and analysis
- **SharePoint Storage**: Generated files automatically saved to SharePoint

### 🔍 Research Capabilities
- **Internal Knowledge Search**: Company expertise and past projects
- **Web Search Integration**: External market research via Tavily

## 📁 Project Structure

```
ba-rfpapp/
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── backend/               # FastAPI REST API
│   ├── api.py            # Main API endpoints
│   └── error_handler.py  # Error handling system
├── frontend/             # React TypeScript App
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── utils/        # API client & utilities
│   │   └── types/        # TypeScript definitions
│   └── package.json
└── ba-server/            # MCP Server
    └── server.py         # Main MCP server
```

## 🔌 API Endpoints

### Core Functionality
- `POST /api/initialize` - Initialize MCP server (tools & prompts)
- `POST /api/chat` - Send chat messages
- `POST /api/upload` - Upload files
- `WS /ws/chat` - WebSocket for real-time chat

### Email & SharePoint
- `POST /api/fetch-emails` - Fetch RFP/RFI/RFQ emails
- `GET /api/fetched-emails` - List fetched emails
- `GET /api/email-attachments/{id}` - Get email attachments
- `GET /api/files` - List SharePoint files
- `GET /api/files/{path}` - Browse SharePoint folders

### System
- `GET /health` - Health check
- `GET /api/status` - Connection status
- `GET /api/tokens` - Token usage statistics

## 🛠️ Available Tools

1. **Broadaxis_knowledge_search** - Internal company knowledge
2. **web_search_tool** - External web search
3. **generate_pdf_document** - Professional PDF creation (saved to SharePoint)
4. **generate_word_document** - Word document generation (saved to SharePoint)
5. **generate_text_file** - Text file creation (saved to SharePoint)

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

- API key management via environment variables
- CORS configuration for frontend access
- File upload validation and sanitization
- Rate limiting and token management
- Error handling and logging system

## 🆘 Troubleshooting

### Common Issues
1. **Email fetching fails**: Check Microsoft Graph API credentials
2. **SharePoint access denied**: Verify site permissions
3. **MCP server offline**: Check Python dependencies
4. **Frontend build errors**: Run `npm install` in frontend directory

### Logs
- Backend logs: Console output
- MCP server logs: `ba-server/mcp_server.log`
- Frontend logs: Browser console

## 📝 License

Proprietary - BroadAxis Internal Use Only

---

**Built with ❤️ for BroadAxis RFP/RFQ Management**