# API Documentation

Complete API reference for the LLM-Driven RealEstate Voice Agent.

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [Base URL](#base-url)
- [Common Headers](#common-headers)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
- [WebSocket Support](#websocket-support)
- [Examples](#examples)

## Overview

The Real Estate Voice Agent API provides endpoints for:
- Natural language conversation
- Speech-to-text conversion
- Text-to-speech generation
- Property search and recommendations
- User requirement management
- System health monitoring

**API Version:** 1.0  
**Protocol:** REST  
**Data Format:** JSON  
**Character Encoding:** UTF-8

## Authentication

Currently, the API does not require authentication for demo purposes.

**Production Implementation (Recommended):**
```http
Authorization: Bearer YOUR_API_TOKEN
```

## Base URL

```
Development: http://localhost:5000
Production: https://api.yourdomain.com
```

## Common Headers

### Request Headers
```http
Content-Type: application/json
Accept: application/json
User-Agent: Your-App/1.0
```

### Response Headers
```http
Content-Type: application/json
X-Request-ID: unique-request-id
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Rate Limiting

**Limits:**
- General API: 100 requests per minute per IP
- Audio endpoints: 20 requests per minute per IP
- Search endpoint: 50 requests per minute per IP

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

**Rate Limit Exceeded Response:**
```json
{
  "error": "RateLimitExceeded",
  "message": "Too many requests. Please try again later.",
  "retry_after": 60
}
```

## Error Handling

### Standard Error Response

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "error_code": "ERROR_CODE",
  "details": {
    "field": "additional context"
  },
  "timestamp": "2024-11-22T10:30:00Z",
  "request_id": "req_123456"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created |
| 400 | Bad Request | Invalid input |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Access denied |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| `INVALID_INPUT` | Invalid request parameters |
| `MISSING_FIELD` | Required field missing |
| `PROPERTY_NOT_FOUND` | Property does not exist |
| `AUDIO_PROCESSING_FAILED` | Audio processing error |
| `LLM_ERROR` | Language model error |
| `VECTOR_DB_ERROR` | Vector database error |
| `RATE_LIMIT_EXCEEDED` | Too many requests |

## Endpoints

### 1. Conversation Endpoint

**POST** `/ask`

Start or continue a conversation with the AI agent.

**Request Body:**
```json
{
  "message": "I'm looking for a 3 bedroom house under $500,000",
  "conversation_id": "conv_123456",
  "user_id": "user_abc",
  "context": {
    "previous_requirements": {
      "budget": 450000,
      "location": "downtown"
    }
  }
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message | string | Yes | User's text message |
| conversation_id | string | No | Unique conversation identifier |
| user_id | string | No | User identifier |
| context | object | No | Additional context |

**Response:**
```json
{
  "response": "I found 3 properties matching your criteria...",
  "conversation_id": "conv_123456",
  "next_action": "show_properties",
  "properties": [
    {
      "property_id": "prop_001",
      "price": 485000,
      "bedrooms": 3,
      "bathrooms": 2,
      "sqft_living": 1800,
      "location": "Downtown",
      "match_type": "exact"
    }
  ],
  "requirements_extracted": {
    "budget": 500000,
    "bedrooms": 3
  },
  "timestamp": "2024-11-22T10:30:00Z"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me houses with 3 bedrooms",
    "conversation_id": "conv_123"
  }'
```

---

### 2. Speech-to-Text Endpoint

**POST** `/audio/transcribe`

Convert audio to text using OpenAI Whisper.

**Request:**
- **Content-Type:** `multipart/form-data`
- **File Parameter:** `audio`

**Supported Formats:**
- WAV
- MP3
- M4A
- WEBM
- FLAC

**Max File Size:** 16MB

**Request Example:**
```bash
curl -X POST http://localhost:5000/audio/transcribe \
  -F "audio=@recording.wav"
```

**Response:**
```json
{
  "text": "I need a three bedroom house under five hundred thousand dollars",
  "language": "en",
  "duration": 3.5,
  "confidence": 0.95,
  "timestamp": "2024-11-22T10:30:00Z"
}
```

**Error Response:**
```json
{
  "error": "AudioProcessingError",
  "message": "Unsupported audio format",
  "supported_formats": ["wav", "mp3", "m4a", "webm", "flac"]
}
```

---

### 3. Text-to-Speech Endpoint

**POST** `/audio/tts`

Convert text to speech using OpenAI TTS.

**Request Body:**
```json
{
  "text": "I found 3 properties that match your requirements",
  "voice": "alloy",
  "speed": 1.0
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| text | string | Yes | - | Text to convert |
| voice | string | No | alloy | Voice selection |
| speed | float | No | 1.0 | Speech speed (0.25-4.0) |

**Available Voices:**
- `alloy` - Neutral and balanced
- `echo` - Male, clear
- `fable` - British accent
- `onyx` - Deep male
- `nova` - Female, warm
- `shimmer` - Female, bright

**Response:**
```json
{
  "audio": "base64_encoded_audio_data",
  "format": "mp3",
  "duration": 2.3,
  "size_bytes": 45678,
  "timestamp": "2024-11-22T10:30:00Z"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/audio/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how can I help you?",
    "voice": "nova"
  }' \
  --output response.json
```

---

### 4. System Status Endpoint

**GET** `/status`

Check system health and configuration.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "components": {
    "openai_api": {
      "status": "connected",
      "latency_ms": 150
    },
    "vector_db": {
      "status": "operational",
      "index_size": 5000,
      "dimension": 1536
    },
    "audio_service": {
      "status": "operational"
    }
  },
  "configuration": {
    "model": "gpt-3.5-turbo",
    "embedding_model": "text-embedding-ada-002",
    "max_properties": 10
  },
  "timestamp": "2024-11-22T10:30:00Z"
}
```

---

### 5. Property Summary Endpoint

**GET** `/property_summary`

Get statistics about the property database.

**Response:**
```json
{
  "total_properties": 5000,
  "price_range": {
    "min": 150000,
    "max": 2500000,
    "average": 650000,
    "median": 550000
  },
  "bedrooms": {
    "min": 1,
    "max": 6,
    "distribution": {
      "1": 250,
      "2": 1500,
      "3": 2000,
      "4": 1000,
      "5+": 250
    }
  },
  "locations": {
    "downtown": 1200,
    "suburbs": 2800,
    "waterfront": 500,
    "rural": 500
  },
  "last_updated": "2024-11-20T00:00:00Z"
}
```

---

### 6. User Requirements Endpoint

**GET** `/requirements`

Retrieve saved user requirements.

**Query Parameters:**
- `user_id` (optional): Filter by user ID
- `limit` (optional): Number of results (default: 10)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "requirements": [
    {
      "requirement_id": "req_001",
      "user_id": "user_abc",
      "budget": 500000,
      "bedrooms": 3,
      "bathrooms": 2,
      "location": "downtown",
      "sqft_min": 1500,
      "email": "user@example.com",
      "phone": "+1234567890",
      "created_at": "2024-11-22T10:00:00Z",
      "updated_at": "2024-11-22T10:30:00Z"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

---

### 7. Property Search Endpoint

**POST** `/search/properties`

Search for properties based on criteria.

**Request Body:**
```json
{
  "budget": 500000,
  "bedrooms": 3,
  "bathrooms": 2,
  "sqft_min": 1500,
  "sqft_max": 2500,
  "location": "downtown",
  "search_type": "exact",
  "limit": 10
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| budget | integer | No | Maximum budget |
| bedrooms | integer | No | Number of bedrooms |
| bathrooms | float | No | Number of bathrooms |
| sqft_min | integer | No | Minimum square footage |
| sqft_max | integer | No | Maximum square footage |
| location | string | No | Location/area |
| search_type | string | No | "exact" or "similar" |
| limit | integer | No | Max results (default: 10) |

**Response:**
```json
{
  "properties": [
    {
      "property_id": "prop_001",
      "price": 485000,
      "bedrooms": 3,
      "bathrooms": 2,
      "sqft_living": 1800,
      "yr_built": 1995,
      "condition": 4,
      "view": 3,
      "waterfront": false,
      "yr_renovated": 2010,
      "location": "Downtown",
      "match_score": 0.95,
      "match_type": "exact"
    }
  ],
  "total_found": 15,
  "returned": 10,
  "search_criteria": {
    "budget": 500000,
    "bedrooms": 3
  },
  "execution_time_ms": 45
}
```

---

### 8. Save Requirements Endpoint

**POST** `/requirements`

Save user requirements for follow-up.

**Request Body:**
```json
{
  "user_id": "user_abc",
  "budget": 500000,
  "bedrooms": 3,
  "bathrooms": 2,
  "location": "downtown",
  "contact": {
    "email": "user@example.com",
    "phone": "+1234567890",
    "whatsapp": "+1234567890"
  },
  "preferences": {
    "garage": true,
    "pool": false,
    "garden": true
  }
}
```

**Response:**
```json
{
  "requirement_id": "req_001",
  "status": "saved",
  "message": "Requirements saved successfully",
  "timestamp": "2024-11-22T10:30:00Z"
}
```

## WebSocket Support

### Connection

```javascript
const ws = new WebSocket('ws://localhost:5000/ws/conversation');

ws.onopen = () => {
  console.log('Connected');
  ws.send(JSON.stringify({
    action: 'start',
    user_id: 'user_123'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Response:', data);
};
```

### Messages

**Send Message:**
```json
{
  "action": "message",
  "text": "Show me 3 bedroom houses",
  "conversation_id": "conv_123"
}
```

**Receive Response:**
```json
{
  "type": "response",
  "text": "I found 3 properties...",
  "properties": [],
  "timestamp": "2024-11-22T10:30:00Z"
}
```

## Examples

### Complete Conversation Flow

```javascript
// 1. Start conversation
const response1 = await fetch('/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "I'm looking for a house",
    conversation_id: "conv_123"
  })
});

// 2. Provide requirements
const response2 = await fetch('/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "3 bedrooms under $500,000",
    conversation_id: "conv_123"
  })
});

// 3. View properties
const properties = await response2.json();

// 4. Save requirements
await fetch('/requirements', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    budget: 500000,
    bedrooms: 3,
    contact: {
      email: "user@example.com"
    }
  })
});
```

### Voice Interaction Flow

```javascript
// 1. Record audio
const audioBlob = await recordAudio();

// 2. Transcribe
const formData = new FormData();
formData.append('audio', audioBlob);

const transcription = await fetch('/audio/transcribe', {
  method: 'POST',
  body: formData
});

const { text } = await transcription.json();

// 3. Send to agent
const response = await fetch('/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: text })
});

// 4. Convert response to speech
const agentResponse = await response.json();
const tts = await fetch('/audio/tts', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: agentResponse.response })
});

const { audio } = await tts.json();
playAudio(audio);
```

## SDK Examples

### Python
```python
import requests

class RealEstateClient:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def ask(self, message, conversation_id=None):
        response = requests.post(
            f"{self.base_url}/ask",
            json={
                "message": message,
                "conversation_id": conversation_id
            }
        )
        return response.json()

# Usage
client = RealEstateClient("http://localhost:5000")
result = client.ask("Show me 3 bedroom houses")
print(result)
```

### JavaScript/Node.js
```javascript
class RealEstateClient {
  constructor(baseURL) {
    this.baseURL = baseURL;
  }
  
  async ask(message, conversationId) {
    const response = await fetch(`${this.baseURL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        conversation_id: conversationId
      })
    });
    return response.json();
  }
}

// Usage
const client = new RealEstateClient('http://localhost:5000');
const result = await client.ask('Show me 3 bedroom houses');
console.log(result);
```

---

**API Version:** 1.0  
**Support:** saifulislamsayem19@gmail.com
