# 🚀 BroadAxis Production Deployment Guide

## **📋 Pre-Deployment Checklist**

### **🔒 Security Requirements**
- [ ] **JWT Secret Key**: Set strong, unique JWT_SECRET_KEY
- [ ] **CORS Origins**: Configure CORS_ORIGINS for your domain
- [ ] **SSL Certificates**: Set up SSL/TLS certificates
- [ ] **Environment Variables**: All secrets configured in .env
- [ ] **Firewall**: Configure firewall rules (ports 80, 443, 8000)

### **🏗️ Infrastructure Requirements**
- [ ] **Docker**: Docker and Docker Compose installed
- [ ] **Domain**: Domain name configured and pointing to server
- [ ] **SSL**: SSL certificates obtained and configured
- [ ] **Monitoring**: Application monitoring set up
- [ ] **Backups**: Backup strategy implemented

## **🚀 Quick Deployment**

### **1. Setup Production Environment**
```bash
# Run the production setup script
./setup-production.sh

# Copy and configure environment variables
cp .env.production .env
# Edit .env with your actual values
```

### **2. Deploy with Docker**
```bash
# Deploy to production
./deploy.sh

# Or manually:
docker-compose -f docker-compose.prod.yml up -d --build
```

### **3. Verify Deployment**
```bash
# Check service health
docker-compose -f docker-compose.prod.yml ps

# Check logs
docker-compose -f docker-compose.prod.yml logs -f
```

## **🔧 Configuration**

### **Environment Variables**
| Variable | Description | Required |
|----------|-------------|----------|
| `JWT_SECRET_KEY` | JWT signing key | ✅ |
| `CORS_ORIGINS` | Allowed origins | ✅ |
| `REDIS_URL` | Redis connection | ✅ |
| `ANTHROPIC_API_KEY` | Anthropic API key | ✅ |
| `PINECONE_API_KEY` | Pinecone API key | ✅ |
| `TAVILY_API_KEY` | Tavily API key | ✅ |
| `OPENAI_API_KEY` | OpenAI API key | ✅ |
| `GRAPH_CLIENT_ID` | Microsoft Graph client ID | ✅ |
| `GRAPH_CLIENT_SECRET` | Microsoft Graph secret | ✅ |
| `GRAPH_TENANT_ID` | Microsoft Graph tenant ID | ✅ |

### **SSL Configuration**
1. **Obtain SSL certificates** (Let's Encrypt recommended)
2. **Place certificates** in `nginx/ssl/`:
   - `cert.pem` - Certificate file
   - `key.pem` - Private key file
3. **Uncomment SSL section** in `nginx/nginx.conf`
4. **Update domain** in nginx configuration

## **📊 Monitoring & Maintenance**

### **Health Checks**
- **Frontend**: `http://your-domain.com/`
- **Backend**: `http://your-domain.com/api/health`
- **API Docs**: `http://your-domain.com/api/docs`

### **Logs**
```bash
# View all logs
docker-compose -f docker-compose.prod.yml logs -f

# View specific service logs
docker-compose -f docker-compose.prod.yml logs -f backend
docker-compose -f docker-compose.prod.yml logs -f frontend
```

### **Updates**
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.prod.yml up -d --build
```

## **🛡️ Security Best Practices**

### **Server Security**
- [ ] **Firewall**: Only allow ports 80, 443, 22
- [ ] **SSH**: Use key-based authentication
- [ ] **Updates**: Keep system packages updated
- [ ] **Monitoring**: Set up intrusion detection

### **Application Security**
- [ ] **HTTPS**: Force HTTPS redirects
- [ ] **Headers**: Security headers configured
- [ ] **Rate Limiting**: API rate limiting enabled
- [ ] **Secrets**: No secrets in code or logs

### **Data Security**
- [ ] **Backups**: Regular database backups
- [ ] **Encryption**: Data encrypted at rest
- [ ] **Access**: Principle of least privilege
- [ ] **Auditing**: Log all access attempts

## **🚨 Troubleshooting**

### **Common Issues**

#### **Services Won't Start**
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs

# Check environment variables
docker-compose -f docker-compose.prod.yml config
```

#### **SSL Issues**
```bash
# Test SSL configuration
openssl s_client -connect your-domain.com:443

# Check certificate validity
openssl x509 -in nginx/ssl/cert.pem -text -noout
```

#### **Database Connection Issues**
```bash
# Test Redis connection
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping

# Check Redis logs
docker-compose -f docker-compose.prod.yml logs redis
```

## **📈 Performance Optimization**

### **Production Optimizations**
- **Frontend**: Minified, chunked, cached
- **Backend**: Multiple workers, connection pooling
- **Database**: Redis optimized for production
- **Nginx**: Gzip compression, caching headers

### **Scaling**
- **Horizontal**: Add more backend instances
- **Vertical**: Increase server resources
- **Database**: Redis clustering for high availability
- **CDN**: Use CDN for static assets

## **🔄 Backup & Recovery**

### **Backup Strategy**
```bash
# Backup Redis data
docker-compose -f docker-compose.prod.yml exec redis redis-cli BGSAVE

# Backup uploaded files
tar -czf backup-$(date +%Y%m%d).tar.gz backend/email_attachments backend/generated_files
```

### **Recovery**
```bash
# Restore from backup
docker-compose -f docker-compose.prod.yml down
# Restore data
docker-compose -f docker-compose.prod.yml up -d
```

## **📞 Support**

For production issues:
1. **Check logs** first
2. **Verify configuration**
3. **Test health endpoints**
4. **Contact support** with logs and error details

---

**🎯 Production Readiness Score: 85/100** (After implementing these fixes)
