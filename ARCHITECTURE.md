"""
# System Architecture

## Overview

The Real Estate Assistant is a modular, microservices-oriented application built with FastAPI, LangChain, and OpenAI.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                         │
│         (Web Browser, Mobile App, API Clients)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway                            │
│          (FastAPI + CORS + Rate Limiting)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Chat Routes │  │ Audio Routes │  │ Admin Routes │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Agent Service│  │ Audio Service│  │ Data Service │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │               │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐       │
│  │Vector Service│  │  OpenAI API  │  │   CSV Data   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ FAISS Vector │  │ Property CSV │  │  User Data   │       │
│  │   Database   │  │   Database   │  │     CSV      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Core Layer (`src/core/`)

**Purpose**: Foundation services and configuration

- `config.py`: Centralized configuration with Pydantic
- `exceptions.py`: Custom exception hierarchy

**Key Features**:
- Environment variable validation
- Type-safe settings
- Singleton pattern for config

### 2. Models Layer (`src/models/`)

**Purpose**: Data structures and validation

- `schemas.py`: Request/response models
- `property.py`: Property domain model

**Key Features**:
- Pydantic validation
- Type hints
- Automatic documentation

### 3. Services Layer (`src/services/`)

**Purpose**: Business logic implementation

#### Data Service
- Loads CSV datasets
- Preprocesses property data
- Handles data cleaning

#### Vector Service
- Manages FAISS vector store
- Creates embeddings
- Handles semantic search

#### Agent Service
- Orchestrates conversation flow
- Manages property search
- Handles requirements extraction
- Formats responses

#### Audio Service
- Transcribes audio (Whisper)
- Synthesizes speech (TTS)
- Handles audio file operations

### 4. API Layer (`src/api/`)

**Purpose**: HTTP endpoints and routing

- **Routes**: REST API endpoints
- **Dependencies**: Dependency injection
- **Middleware**: CORS, logging, etc.

### 5. Utils Layer (`src/utils/`)

**Purpose**: Helper functions

- **Formatters**: Property display logic
- **Validators**: Data validation utilities

## Data Flow

### Chat Request Flow

```
1. Client sends message
   ↓
2. FastAPI receives request
   ↓
3. Chat route validates input
   ↓
4. Agent Service processes:
   a. Extract requirements
   b. Search vector store
   c. Match properties
   d. Generate response
   ↓
5. Response sent to client
```

### Audio Flow

```
1. Client uploads audio
   ↓
2. Audio route receives file
   ↓
3. Audio Service transcribes
   ↓
4. Text sent to Agent Service
   ↓
5. Response synthesized to speech
   ↓
6. Audio returned to client
```

## Design Patterns

### 1. Dependency Injection
- Services injected via FastAPI dependencies
- Promotes testability
- Reduces coupling

### 2. Service Layer Pattern
- Business logic separated from API
- Reusable services
- Clear separation of concerns

### 3. Repository Pattern
- Data access abstracted
- Easy to swap data sources
- Consistent interface

### 4. Factory Pattern
- Document creation
- Property formatting
- Service initialization

## Technology Stack

| Layer | Technology |
|-------|------------|
| Web Framework | FastAPI |
| AI/ML | LangChain, OpenAI |
| Vector DB | FAISS |
| Data Processing | Pandas, NumPy |
| Validation | Pydantic |
| Testing | Pytest |
| Logging | Python logging |

## Performance Considerations

### Caching Strategy
- Vector store cached by file hash
- Embeddings cached to disk
- LLM responses not cached (dynamic)

### Optimization Techniques
- Batch processing for embeddings
- Async/await for I/O operations
- Lazy loading of services
- Connection pooling (if using DB)

## Security Measures

- API key validation on startup
- Environment variable protection
- Input sanitization (Pydantic)
- CORS configuration
- Rate limiting (recommended)

## Monitoring & Observability

- Structured logging
- Health check endpoints
- Status monitoring
- Error tracking
- Performance metrics

## Future Enhancements

1. **Database Integration**: PostgreSQL for persistent storage
2. **Caching Layer**: Redis for session management
3. **Message Queue**: Celery for async tasks
4. **Authentication**: OAuth2 for user management
5. **Analytics**: Property view tracking
6. **Notifications**: Email/SMS alerts
7. **Multi-language**: i18n support
8. **Advanced Search**: Elasticsearch integration
"""
