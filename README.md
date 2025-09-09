# BroadAxis RFP Management & Trading Platform

A comprehensive full-stack application combining RFP (Request for Proposal) management with advanced trading analysis capabilities, powered by AI and modern web technologies.

## 🏗️ Architecture Overview

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS with custom glassmorphism design
- **State Management**: React Context API + useState/useEffect
- **Build Tool**: Vite
- **UI Components**: Custom components with modern design patterns
- **Real-time Communication**: WebSocket integration for live updates

### Backend (FastAPI + Python)
- **Framework**: FastAPI with async/await support
- **Authentication**: JWT-based with bcrypt password hashing
- **Session Management**: Redis-based session storage
- **AI Integration**: 
  - Anthropic Claude for RFP analysis
  - OpenAI for embeddings and text processing
  - Custom MCP (Model Context Protocol) interface
- **Database**: Redis for sessions and caching
- **Vector Database**: Pinecone for document embeddings
- **File Processing**: PDF, DOCX, and image processing capabilities

### Key Features

#### 🔍 RFP Management System
- **Document Processing**: Automated PDF/DOCX parsing and analysis
- **AI-Powered Analysis**: Intelligent RFP evaluation and scoring
- **Email Integration**: Automated email processing with attachment handling
- **SharePoint Integration**: Enterprise document management
- **Real-time Collaboration**: WebSocket-based live updates
- **Advanced Search**: Semantic search using vector embeddings

#### 📈 Trading Planner
- **AI Trading Analysis**: Claude-powered trading insights and recommendations
- **Session Management**: Redis-backed trading session persistence
- **Advanced Charting**: Interactive trading data visualization
- **Risk Assessment**: Automated risk analysis and scoring
- **Portfolio Management**: Multi-session trading strategy tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Node.js 18+
- Redis server
- API keys for OpenAI, Anthropic, Pinecone, and Tavily

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Broadaxis-frontend-backend
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

3. **Install Node.js dependencies**
   ```bash
   cd frontend
   npm install
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```bash
   # Database Configuration
   REDIS_URL=your-redis-connection-string
   
   # JWT Configuration
   JWT_SECRET_KEY=your-super-secret-jwt-key
   
   # API Keys
   OPENAI_API_KEY=your-openai-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   TAVILY_API_KEY=your-tavily-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_ENVIRONMENT=your-pinecone-environment
   
   # CORS Configuration (for production)
   # CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com
   
   # Environment
   ENVIRONMENT=development
   ```

### Running the Application

1. **Start the backend**
   ```bash
   cd backend
   python run_backend.py
   ```
   Backend will be available at: http://localhost:8000

2. **Start the frontend**
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend will be available at: http://localhost:3000

## 🔧 Development

### Project Structure
```
Broadaxis-frontend-backend/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/          # Main application pages
│   │   ├── context/        # React context providers
│   │   ├── utils/          # Utility functions and API client
│   │   └── types/          # TypeScript type definitions
│   ├── dist/               # Built frontend assets
│   └── package.json        # Frontend dependencies
├── backend/                 # FastAPI backend application
│   ├── api.py              # Main API endpoints
│   ├── session_manager.py  # Redis session management
│   ├── mcp_interface.py    # MCP protocol implementation
│   ├── websocket_api.py    # WebSocket handlers
│   ├── email_api.py        # Email processing endpoints
│   ├── sharepoint_api.py   # SharePoint integration
│   └── token_manager.py    # JWT token management
├── embedding.py            # Document embedding utilities
├── requirements.txt        # Python dependencies
└── .env                    # Environment configuration
```

### Key Components

#### Frontend Components
- **AuthContext**: User authentication and session management
- **AppContext**: Global application state and WebSocket connection
- **ChatInterface**: Main RFP management chat interface
- **TradingPlanner**: Advanced trading analysis interface
- **DashboardLayout**: Main application layout with sidebar
- **AutoExpandingTextarea**: Dynamic input field for long messages

#### Backend Services
- **Session Manager**: Redis-based user session handling
- **MCP Interface**: Model Context Protocol for AI integration
- **WebSocket API**: Real-time communication endpoints
- **Email API**: Automated email processing and attachment handling
- **SharePoint API**: Enterprise document management integration
- **Token Manager**: JWT token generation and validation

### API Endpoints

#### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/forgot` - Password reset request
- `POST /api/auth/reset` - Password reset confirmation

#### RFP Management
- `POST /api/chat` - Main chat interface
- `GET /api/sessions` - Get user sessions
- `POST /api/sessions` - Create new session
- `DELETE /api/sessions/{id}` - Delete session
- `POST /api/upload` - File upload and processing

#### Trading Planner
- `POST /api/trading/chat` - Trading analysis chat
- `GET /api/trading/sessions` - Get trading sessions
- `POST /api/trading/session/create` - Create trading session
- `DELETE /api/trading/session/{id}` - Delete trading session

#### WebSocket
- `ws://localhost:8000/ws/chat` - Real-time chat updates

## 🚀 Production Deployment

### Docker Deployment

1. **Build and deploy with Docker Compose**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Configure production environment**
   - Set `CORS_ORIGINS` to your production domains
   - Configure `JWT_SECRET_KEY` with a strong secret
   - Set up SSL certificates in `nginx/ssl/`
   - Configure Redis connection string

### Manual Deployment

1. **Backend deployment**
   ```bash
   cd backend
   pip install -r ../requirements.txt
   gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **Frontend deployment**
   ```bash
   cd frontend
   npm run build
   # Serve dist/ directory with nginx or similar
   ```

3. **Nginx configuration**
   Use the provided `nginx/nginx.conf` as a starting point for reverse proxy setup.

### Environment Configuration

#### Development
- CORS allows localhost origins (3000, 5173)
- Development JWT secret (change for production)
- Debug logging enabled

#### Production
- CORS restricted to specific domains
- Strong JWT secret required
- Security headers enabled
- Rate limiting configured

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt for secure password storage
- **CORS Protection**: Configurable cross-origin resource sharing
- **Rate Limiting**: API rate limiting to prevent abuse
- **Input Validation**: Pydantic models for request validation
- **Environment Variables**: Sensitive data stored in environment variables
- **Redis Security**: Secure Redis connection with authentication

## 🧪 Testing

### Backend Testing
```bash
cd backend
python -m pytest tests/
```

### Frontend Testing
```bash
cd frontend
npm test
```

## 📊 Monitoring & Logging

- **Structured Logging**: JSON-formatted logs for production
- **Error Tracking**: Comprehensive error handling and reporting
- **Performance Monitoring**: Request timing and resource usage
- **Health Checks**: `/api/status` endpoint for service health

## 🔄 CI/CD Pipeline

The application supports automated deployment with:
- Docker containerization
- Environment-specific configurations
- Automated testing
- Production deployment scripts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is proprietary software. All rights reserved.

## 🆘 Support

For technical support or questions:
- Check the API documentation at `/docs` when running locally
- Review the logs in `backend/mcp_server.log`
- Ensure all environment variables are properly configured
- Verify Redis connection and API key validity

## 🔄 Recent Updates

- **v1.0.0**: Initial release with RFP management and trading planner
- **v1.1.0**: Added WebSocket real-time updates
- **v1.2.0**: Enhanced trading analysis with AI insights
- **v1.3.0**: Production-ready deployment configuration
- **v1.4.0**: Advanced document processing and SharePoint integration

---

**Built with ❤️ by the BroadAxis Team**
