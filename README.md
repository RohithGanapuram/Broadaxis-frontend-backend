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
- uv (optional, for faster Python package management)

### Quick Installation
```bash
# Install all dependencies at once
python install_dependencies.py
```

### Environment Setup
1. Create `.env` file in `ba-server/` directory:
```env
ANTHROPIC_API_KEY=your_anthropic_key
PINECONE_API_KEY=your_pinecone_key
TAVILY_API_KEY=your_tavily_key
```

### Option 1: Run Everything at Once
```bash
python start_all.py
```

### Option 2: Run Services Separately

#### Backend (FastAPI)
```bash
cd backend
python run_backend.py
# Access: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

#### Frontend (React)
```bash
cd frontend
npm install
npm run dev
# Access: http://localhost:3000
```

#### MCP Server (Optional - for testing)
```bash
cd ba-server
python run.py
# Access: http://localhost:8503
```

## 🔧 Key Features

### 🤖 AI-Powered Analysis
- **Automatic RFP/RFQ Processing**: Upload documents and get instant analysis
- **Go/No-Go Recommendations**: AI-driven decision support based on company capabilities
- **Company Knowledge Integration**: Leverages internal knowledge base via Pinecone

### 📄 Document Management
- **Multi-format Support**: PDF, DOCX, DOC, TXT, MD files
- **Real-time Processing**: Instant text extraction and analysis
- **Generated Files**: Create professional PDFs, Word docs, and text files

### 🔍 Research Capabilities
- **Internal Knowledge Search**: Company expertise and past projects
- **Web Search Integration**: External market research via Tavily
- **Academic Research**: arXiv paper search and analysis

### 🛠️ Available Tools
1. **Broadaxis_knowledge_search** - Internal company knowledge
2. **web_search_tool** - External web search
3. **generate_pdf_document** - Professional PDF creation
4. **generate_word_document** - Word document generation
5. **generate_text_file** - Text file creation
6. **search_papers** - Academic research
7. **Weather tools** - Project planning support

## 📁 Project Structure

```
ba-rfpapp/
├── backend/                 # FastAPI REST API
│   ├── api.py              # Main API endpoints
│   └── run_backend.py      # Backend launcher
├── frontend/               # React TypeScript App
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── utils/          # API client & utilities
│   │   └── types/          # TypeScript definitions
│   ├── package.json
│   └── vite.config.ts
├── ba-server/              # MCP Server & Streamlit
│   ├── server.py           # Main MCP server
│   ├── streamlit_ui.py     # Streamlit interface
│   ├── file_server.py      # File management utilities
│   └── pyproject.toml      # Python dependencies
└── start_all.py           # Startup script
```

## 🔌 API Endpoints

### Chat & Analysis
- `POST /api/chat` - Send chat messages
- `POST /api/upload` - Upload files
- `POST /api/upload-and-analyze` - Upload and auto-analyze
- `WS /ws/chat` - WebSocket for real-time chat

### Tools & Configuration
- `GET /api/tools` - List available MCP tools
- `GET /api/prompts` - List available prompts
- `GET /health` - Health check

### File Management
- `GET /api/files` - List generated files
- `GET /api/files/{filename}` - Download file
- `DELETE /api/files/{filename}` - Delete file
- `POST /api/files/cleanup` - Clean up old files

## 🎯 Usage Workflow

### 1. RFP/RFQ Analysis
1. Upload RFP document via drag & drop
2. Get automatic analysis with key requirements
3. Request Go/No-Go recommendation
4. Generate response documents

### 2. Company Knowledge Search
1. Ask questions about company capabilities
2. Search past projects and experience
3. Get team member information
4. Access internal processes

### 3. Document Generation
1. Create professional PDFs and Word docs
2. Generate executive summaries
3. Build compliance matrices
4. Export research findings

## 🔧 Development

### Dependency Management
- **Consolidated**: Single `requirements.txt` at project root
- **ba-server**: Uses `pyproject.toml` with uv for faster installs
- **Frontend**: Standard `package.json` with npm/yarn

### Backend Development
```bash
# Install dependencies (from project root)
pip install -r requirements.txt
# Run backend
cd backend
python run_backend.py
```

### Frontend Development
```bash
cd frontend
npm install  # or yarn install
npm run dev  # or yarn dev
```

### MCP Server Development
```bash
cd ba-server
# With uv (recommended)
uv sync
# Or with pip
pip install -e .
# Test server
python client.py
```

## 🚀 Deployment

### Docker (Coming Soon)
```bash
docker-compose up
```

### Manual Deployment
1. Build React frontend: `npm run build`
2. Deploy FastAPI with gunicorn
3. Configure reverse proxy (nginx)
4. Set up environment variables

## 🔐 Security

- API key management via environment variables
- CORS configuration for frontend access
- File upload validation and sanitization
- Rate limiting on API endpoints

## 📊 Monitoring

- Health check endpoint: `/health`
- Server status in React settings
- WebSocket connection monitoring
- File generation tracking

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📝 License

Proprietary - BroadAxis Internal Use Only

## 🆘 Support

For issues and questions:
1. Check API documentation: `http://localhost:8000/docs`
2. Review server logs
3. Test MCP server connection
4. Verify environment variables

---

**Built with ❤️ for BroadAxis RFP/RFQ Management**