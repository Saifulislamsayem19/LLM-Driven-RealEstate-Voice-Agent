# System Architecture

## Table of Contents
- [Overview](#overview)
- [System Design](#system-design)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Design Patterns](#design-patterns)
- [Scalability Considerations](#scalability-considerations)
- [Security Architecture](#security-architecture)

## Overview

The LLM-Driven RealEstate Voice Agent is a sophisticated conversational AI system that combines natural language processing, speech recognition/synthesis, and semantic search to provide an intelligent property search experience.

### Core Principles
- **Modularity**: Each component has a single, well-defined responsibility
- **Extensibility**: New features can be added without modifying existing code
- **Reliability**: Comprehensive error handling and fallback mechanisms
- **Performance**: Optimized for low-latency responses
- **Maintainability**: Clean code structure with comprehensive documentation

## System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (Web Browser / Mobile)                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   /ask       │  │  /transcribe │  │    /tts      │           │
│  │   endpoint   │  │   endpoint   │  │   endpoint   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LangGraph Workflow Engine                   │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │   │
│  │  │  Intent    │→ │  Retrieval │→ │  Response  │          │   │
│  │  │ Analysis   │  │    Node    │  │ Generation │          │   │
│  │  └────────────┘  └────────────┘  └────────────┘          │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 Audio Processing                         │   │
│  │          ┌──────────┐        ┌──────────┐                │   │
│  │          │   STT    │        │   TTS    │                │   │
│  │          │ (Whisper)│        │ (OpenAI) │                │   │
│  │          └──────────┘        └──────────┘                │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│  │  Vector DB │  │ Property   │  │   User     │                 │ 
│  │  (FAISS)   │  │   Data     │  │Requirements│                 │
│  └────────────┘  └────────────┘  └────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Diagram

```
User Query (Voice/Text)
         │
         ▼
    ┌─────────┐
    │   API   │
    └────┬────┘
         │
         ▼
    ┌─────────────────┐
    │ Audio Processor │ (if voice)
    └────┬────────────┘
         │
         ▼
    ┌─────────────────────┐
    │  LangGraph Agent    │
    └────┬────────────────┘
         │
         ├──▶ Intent Classification
         │
         ├──▶ Requirement Extraction
         │         │
         │         ▼
         │    ┌────────────┐
         │    │ Vector DB  │
         │    │  Embedder  │
         │    └─────┬──────┘
         │          │
         ├──────────┘
         │
         ├──▶ Property Search (FAISS)
         │         │
         │         ▼
         │    ┌────────────┐
         │    │ Exact/     │
         │    │ Similar    │
         │    │ Match      │
         │    └─────┬──────┘
         │          │
         ├──────────┘
         │
         ├──▶ Response Generation (LLM)
         │
         ▼
    ┌─────────────────┐
    │  TTS (if voice) │
    └────┬────────────┘
         │
         ▼
    User Response
```

## Component Architecture

### 1. API Layer (app.py)

**Responsibilities:**
- HTTP request handling
- Route management
- Request validation
- Response formatting
- CORS handling

**Key Endpoints:**
- `POST /ask` - Main conversation endpoint
- `POST /audio/transcribe` - Speech-to-text
- `POST /audio/tts` - Text-to-speech
- `GET /status` - System health check
- `GET /property_summary` - Property statistics

**Design Pattern:** RESTful API with Flask/FastAPI

### 2. AI Agent (ai_agent.py)

**Responsibilities:**
- Conversation state management
- Workflow orchestration
- Intent classification
- Requirement extraction
- Decision making logic

**Core Components:**

```python
AgentState (TypedDict):
    - messages: List[BaseMessage]
    - requirements: Dict
    - search_results: List[Dict]
    - next_action: str
    - contact_collected: bool
```

**State Graph Nodes:**
- `retrieve_documents`: Vector search execution
- `extract_requirements`: Parse user criteria
- `decide_next_action`: Workflow routing
- `collect_contact`: User info gathering
- `generate_response`: LLM response generation

**Design Pattern:** State Machine with LangGraph

### 3. Audio Processing (audio.py)

**Responsibilities:**
- Audio format conversion
- Speech recognition
- Voice synthesis
- Audio streaming

**Components:**
- **STT Engine**: OpenAI Whisper API
- **TTS Engine**: OpenAI TTS API
- **Audio Preprocessor**: Format conversion, noise reduction

**Supported Formats:**
- Input: WAV, MP3, M4A, WEBM
- Output: MP3 (base64 encoded)

**Design Pattern:** Adapter Pattern for different audio formats

### 4. Data Manager (data_manager.py)

**Responsibilities:**
- Property data loading
- User requirements storage
- Data validation
- CSV operations
- Data transformation

**Key Functions:**
- `load_properties()`: Load property dataset
- `save_user_requirements()`: Persist user data
- `get_property_statistics()`: Analytics

**Design Pattern:** Repository Pattern

### 5. Vector Database

**Technology:** FAISS (Facebook AI Similarity Search)

**Architecture:**
```
Property Data
     │
     ▼
Text Embedding (OpenAI ada-002)
     │
     ▼
Vector Storage (FAISS Index)
     │
     ▼
Similarity Search
     │
     ▼
Ranked Results
```

**Index Structure:**
- Dimension: 1536 (OpenAI embedding size)
- Index Type: Flat (exact search)
- Metric: Cosine similarity

**Design Pattern:** Vector Store with embedding cache

## Data Flow

### 1. Voice Query Flow

```
1. User speaks → Audio captured (WebRTC)
2. Audio sent to /audio/transcribe
3. Whisper API processes audio
4. Text returned to frontend
5. Text sent to /ask endpoint
6. Agent processes query
7. Response generated
8. Text sent to /audio/tts
9. TTS generates speech
10. Audio returned to user
```

### 2. Property Search Flow

```
1. User requirements extracted from query
2. Requirements converted to embedding
3. FAISS vector search executed
4. Top K similar properties retrieved
5. Exact match filtering applied
6. Similar properties identified (±10% budget)
7. Results ranked by relevance
8. LLM generates personalized response
9. Response returned to user
```

### 3. RAG (Retrieval-Augmented Generation) Flow

```
Query → Embedding → Vector Search → Context Retrieval
                                          │
                                          ▼
User Query + Retrieved Context → LLM → Enhanced Response
```

## Technology Stack

### Backend Framework
- **FastAPI/Flask**: Web framework
- **Python 3.8+**: Programming language

### AI/ML Stack
- **LangChain**: LLM orchestration framework
- **LangGraph**: Workflow state management
- **OpenAI GPT-3.5/4**: Language model
- **OpenAI Whisper**: Speech recognition
- **OpenAI TTS**: Speech synthesis
- **FAISS**: Vector similarity search

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Data preprocessing

### Infrastructure
- **Docker**: Containerization (recommended)
- **Gunicorn/Uvicorn**: WSGI/ASGI server
- **Redis**: Caching (optional)

## Design Patterns

### 1. State Machine Pattern
**Where:** LangGraph workflow
**Why:** Manage complex conversation flows with multiple states

### 2. Repository Pattern
**Where:** Data management layer
**Why:** Abstract data access logic from business logic

### 3. Strategy Pattern
**Where:** Search algorithms (exact vs similar match)
**Why:** Switch between different search strategies dynamically

### 4. Adapter Pattern
**Where:** Audio processing
**Why:** Support multiple audio formats uniformly

### 5. Singleton Pattern
**Where:** Configuration management
**Why:** Single source of truth for application settings

### 6. Chain of Responsibility
**Where:** Request processing pipeline
**Why:** Process requests through multiple handlers

## Scalability Considerations

### Current Limitations
- Single-threaded audio processing
- In-memory vector storage
- No horizontal scaling support
- Synchronous API calls

### Recommended Improvements

**1. Horizontal Scaling**
```
Load Balancer
     │
     ├─▶ API Instance 1
     ├─▶ API Instance 2
     └─▶ API Instance 3
          │
          ▼
    Shared Vector DB
    (Pinecone/Weaviate)
```

**2. Asynchronous Processing**
- Use async/await for I/O operations
- Implement message queues (RabbitMQ/Celery)
- Background task processing

**3. Caching Strategy**
- Redis for embedding cache
- Response caching for common queries
- CDN for static assets

**4. Database Optimization**
- Use production vector database (Pinecone, Weaviate)
- Implement database connection pooling
- Add read replicas for queries

**5. Microservices Architecture**
```
API Gateway
     │
     ├─▶ Agent Service
     ├─▶ Audio Service
     ├─▶ Search Service
     └─▶ Data Service
```

## Security Architecture

### Current Implementation
- Environment variable configuration
- No authentication/authorization
- No rate limiting
- No input sanitization

### Recommended Security Measures

**1. Authentication & Authorization**
- JWT token-based authentication
- Role-based access control (RBAC)
- API key management

**2. Data Protection**
- Encrypt sensitive data at rest
- Use HTTPS/TLS for data in transit
- Implement data retention policies

**3. Input Validation**
- Sanitize all user inputs
- Validate file uploads
- Implement rate limiting

**4. API Security**
- CORS configuration
- Request size limits
- DDoS protection
- API versioning

**5. Monitoring & Logging**
- Security event logging
- Anomaly detection
- Audit trails
- Error tracking (Sentry)

### Security Headers
```python
{
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000",
    "Content-Security-Policy": "default-src 'self'"
}
```

## Performance Optimization

### Current Bottlenecks
1. OpenAI API latency (200-2000ms)
2. Vector embedding generation
3. Audio file processing
4. Synchronous blocking calls

### Optimization Strategies

**1. Caching**
- Cache embeddings for common queries
- Cache LLM responses for identical inputs
- Implement CDN for static content

**2. Batch Processing**
- Batch embedding generation
- Parallel property searches
- Concurrent API calls

**3. Database Optimization**
- Index optimization
- Query result pagination
- Lazy loading

**4. Code Optimization**
- Profile hotspots
- Optimize loops
- Use generators for large datasets

## Monitoring & Observability

### Recommended Metrics

**System Metrics:**
- CPU/Memory usage
- Request latency
- Error rates
- API call counts

**Business Metrics:**
- Successful property matches
- User query types
- Conversation completion rate
- Audio processing success rate

**Monitoring Stack:**
- Prometheus: Metrics collection
- Grafana: Visualization
- ELK Stack: Log aggregation
- Sentry: Error tracking

## Future Enhancements

1. **Multi-language Support**
2. **Image Recognition** (property photos)
3. **Advanced Filtering** (school districts, crime rates)
4. **Recommendation Engine** (ML-based)
5. **Real-time Collaboration** (WebSocket)
6. **Mobile App** (React Native)
7. **Calendar Integration** (property viewings)
8. **Payment Integration** (deposits)
