#!/bin/bash

# BroadAxis Production Setup Script
echo "🚀 Setting up BroadAxis for Production..."

# Create production environment file
echo "📝 Creating production environment template..."
cat > .env.production << 'EOF'
# Production Environment Variables
# Copy this to .env and fill in your actual values

# Security
JWT_SECRET_KEY=your-production-jwt-secret-key-here
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com

# Redis Configuration
REDIS_URL=redis://redis:6379

# AI Services
ANTHROPIC_API_KEY=your_anthropic_key
PINECONE_API_KEY=your_pinecone_key
TAVILY_API_KEY=your_tavily_key
OPENAI_API_KEY=your_openai_key

# Microsoft Graph API
GRAPH_CLIENT_ID=your_client_id
GRAPH_CLIENT_SECRET=your_client_secret
GRAPH_TENANT_ID=your_tenant_id

# Email Accounts
GRAPH_USER_EMAIL_1=email1@company.com
GRAPH_USER_EMAIL_2=email2@company.com
GRAPH_USER_EMAIL_3=email3@company.com

# Trading Planner Access
TRADING_ALLOWED_EMAILS=admin@company.com,trader@company.com

# SharePoint Configuration
SHAREPOINT_SITE_URL=company.sharepoint.com:/sites/project
SHAREPOINT_FOLDER_PATH=Documents
EOF

echo "✅ Production environment template created!"

# Create SSL directory
echo "🔒 Creating SSL directory..."
mkdir -p nginx/ssl

# Create production deployment script
echo "📦 Creating deployment script..."
cat > deploy.sh << 'EOF'
#!/bin/bash

# BroadAxis Production Deployment Script
echo "🚀 Deploying BroadAxis to Production..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found! Please copy .env.production to .env and configure it."
    exit 1
fi

# Build and start services
echo "🏗️ Building and starting services..."
docker-compose -f docker-compose.prod.yml up -d --build

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check health
echo "🔍 Checking service health..."
docker-compose -f docker-compose.prod.yml ps

echo "✅ Deployment complete!"
echo "🌐 Frontend: http://localhost"
echo "🔧 Backend: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
EOF

chmod +x deploy.sh

echo "✅ Production setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy .env.production to .env and configure your values"
echo "2. Run ./deploy.sh to deploy to production"
echo "3. Configure SSL certificates in nginx/ssl/ if needed"
echo ""
echo "🔒 Security checklist:"
echo "- [ ] Set strong JWT_SECRET_KEY"
echo "- [ ] Configure CORS_ORIGINS for your domain"
echo "- [ ] Set up SSL certificates"
echo "- [ ] Configure firewall rules"
echo "- [ ] Set up monitoring"
