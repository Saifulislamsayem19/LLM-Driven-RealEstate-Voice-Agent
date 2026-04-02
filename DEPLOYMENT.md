"""
# Deployment Guide

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd real-estate-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run Application

```bash
# Development mode
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

### 4. Access Application

- **Web Interface**: http://localhost:7860
- **API Docs**: http://localhost:7860/docs
- **Admin Panel**: http://localhost:7860/api-admin

## Production Deployment

### Using Docker

```bash
# Build image
docker build -t real-estate-assistant .

# Run container
docker run -d \
  -p 7860:7860 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/vector_db:/app/vector_db \
  --name real-estate-app \
  real-estate-assistant
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Systemd (Linux)

Create `/etc/systemd/system/real-estate.service`:

```ini
[Unit]
Description=Real Estate Assistant
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/real-estate-assistant
Environment="PATH=/opt/real-estate-assistant/venv/bin"
ExecStart=/opt/real-estate-assistant/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable real-estate
sudo systemctl start real-estate
sudo systemctl status real-estate
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENAI_API_KEY | OpenAI API key (required) | - |
| MODEL_NAME | LLM model to use | gpt-3.5-turbo-16k |
| TEMPERATURE | Model temperature | 0.3 |
| DEBUG | Enable debug mode | False |
| HOST | Server host | 0.0.0.0 |
| PORT | Server port | 7860 |
| LOG_LEVEL | Logging level | INFO |

## Monitoring

### Health Check

```bash
curl http://localhost:7860/health
```

### Logs

```bash
# View application logs
tail -f logs/app.log

# Docker logs
docker logs -f real-estate-app
```

### Metrics

Access `/status` endpoint for system metrics:

```bash
curl http://localhost:7860/status
```

## Troubleshooting

### Common Issues

**1. Import Errors**

```bash
# Ensure you're in the project root and venv is activated
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. Vector Store Issues**

```bash
# Clear vector database cache
rm -rf vector_db/*
# Restart application to rebuild
```

**3. API Key Issues**

- Check `.env` file has correct key
- Verify key has sufficient credits
- Check key permissions

## Security

### Production Checklist

- [ ] Use HTTPS (reverse proxy with nginx/caddy)
- [ ] Set DEBUG=False
- [ ] Use strong API keys
- [ ] Enable rate limiting
- [ ] Configure CORS appropriately
- [ ] Set up monitoring/alerting
- [ ] Regular backups of user_data/
- [ ] Keep dependencies updated

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Scaling

### Horizontal Scaling

Use load balancer with multiple instances:

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    image: real-estate-assistant
    deploy:
      replicas: 3
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Vertical Scaling

Increase resources per instance:

```bash
# Docker resource limits
docker run -d \
  --cpus="2" \
  --memory="4g" \
  real-estate-assistant
```
"""
