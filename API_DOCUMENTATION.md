# API Documentation

API reference for the LLM-Driven RealEstate Voice Agent (HomeFind AI). Everything here matches the code in
`app/api/routes/`. Interactive docs generated from the code are also served at `/docs` (Swagger) and `/redoc`.

## Table of Contents
- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Limits](#limits)
- [Errors](#errors)
- [Endpoints](#endpoints)
- [Examples](#examples)

## Overview

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/` | GET | – | Web chat UI |
| `/ask` | POST | – | Send a chat message, get the assistant's reply |
| `/audio/transcribe` | POST | – | Speech-to-text |
| `/audio/tts` | POST | – | Text-to-speech |
| `/status` | GET | – | Readiness and active AI providers |
| `/property_summary` | GET | – | Dataset statistics |
| `/requirements` | GET | Admin | Saved user requirements and contact details |
| `/api-@dmin` | GET, POST | Admin | Admin panel for updating the OpenAI key |

**AI providers:** OpenAI is used while its key works. If it is missing, invalid or out of quota, chat and speech
fall back to Groq and embeddings to a local model. `GET /status` shows which is active.

## Base URL

```
http://localhost:7860
```

On Hugging Face Spaces, use the Space's `https://<user>-<space>.hf.space` URL.

## Authentication

Public endpoints need no authentication. The web UI calls them from the same origin; cross-origin browser
access is disabled unless `ALLOWED_ORIGINS` is set.

Admin endpoints use **HTTP Basic** auth:

- Username: `ADMIN_USERNAME` (default `admin`)
- Password: `ADMIN_PASSWORD`
- While `ADMIN_PASSWORD` is empty, admin endpoints return `404` (disabled).
- Wrong or missing credentials return `401`.
- `POST /api-@dmin` also requires an `Origin` or `Referer` header from the same host (blocks cross-site form posts).

## Limits

Rate limits are per client IP, over a sliding 60-second window. They are kept in memory, so they apply per
running instance.

| Limit | Default | Setting |
|---|---|---|
| `/ask` requests per minute | 20 | `RATE_LIMIT_CHAT_PER_MINUTE` (0 = off) |
| `/audio/*` requests per minute | 20 | `RATE_LIMIT_AUDIO_PER_MINUTE` (0 = off) |
| `/ask` messages per day that reach the AI, per IP | 150 | `DAILY_CHAT_LIMIT_PER_IP` (0 = off) |
| `/ask` messages per day that reach the AI, all visitors | 3,000 | `DAILY_CHAT_LIMIT_GLOBAL` (0 = off) |
| Jailbreak attempts before lockout | 5, then locked out for 15 minutes | `JAILBREAK_STRIKE_LIMIT`, `JAILBREAK_LOCKOUT_MINUTES` (0 = off) |
| Message length (`query`) | 2,000 characters | `MAX_QUERY_CHARS` |
| Chat history sent with a message | last 60 entries kept, rest ignored | `MAX_HISTORY_MESSAGES` |
| Text-to-speech input | 4,000 characters (longer text is truncated) | `MAX_TTS_CHARS` |
| Audio upload size | 25 MB | `MAX_AUDIO_UPLOAD_MB` |

When a rate limit is exceeded the response is `429` with a `Retry-After` header (seconds).

## Errors

Errors use FastAPI's standard shape:

```json
{ "detail": "Too many requests — please wait a moment and try again." }
```

`/ask` is the exception: it always returns an `answer` string the UI can display, even on errors.

| Status | Meaning |
|---|---|
| 400 | Malformed request (invalid JSON, wrong field types, empty TTS text) |
| 401 | Admin credentials missing or wrong |
| 403 | Cross-site admin form post blocked |
| 404 | Not found, or admin endpoints disabled (no `ADMIN_PASSWORD`) |
| 413 | Message, or audio upload, too large |
| 415 | Uploaded file is not audio |
| 429 | Rate limit exceeded |
| 500 | Internal error (details are logged on the server, not returned) |
| 502 | Speech provider unavailable (both OpenAI and Groq failed) |
| 503 | `/status` only: still starting up |

## Endpoints

### POST `/ask`

Send a user message and get the assistant's reply.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | The user's message (max 2,000 chars). An empty string returns a greeting. |
| `chat_history` | array | No | Earlier messages, oldest first. See below. |

`chat_history` entries are `{ "role": "user" | "assistant" | "system", "content": string }`. The server is
stateless: to continue a conversation, send back the previous messages **and** the latest `system_state`
as a `system` entry whose `content` is that object serialized as a JSON string. This carries the collected
requirements, shown properties and conversation stage between turns.

```json
{
  "query": "Tell me more about the first one",
  "chat_history": [
    { "role": "user", "content": "3 bedroom house in Seattle under 600000" },
    { "role": "assistant", "content": "Here are three matches..." },
    { "role": "system", "content": "{\"type\":\"system\",\"requirements\":{\"budget\":600000,\"location\":\"Seattle\",\"bedrooms\":3},\"conversation_stage\":\"gathering_requirements\",\"properties_shown\":[0,3,8]}" }
  ]
}
```

**Response** `200`

```json
{
  "answer": "Here are three 3-bedroom houses in Seattle that fit your budget...",
  "system_state": {
    "type": "system",
    "requirements": { "budget": 600000, "location": "Seattle", "bedrooms": 3 },
    "conversation_stage": "gathering_requirements",
    "contact_info": {},
    "properties_shown": [0, 3, 8],
    "exact_matches": [
      { "property_id": 0, "price": 376000.0, "bedrooms": 3.0, "bathrooms": 2.0, "sqft_living": 1340,
        "city": "Seattle", "statezip": "WA 98103", "yr_built": 2008, "...": "..." }
    ],
    "similar_matches": [],
    "needs_contact_info": false,
    "no_properties_found": false,
    "requirements_saved": false,
    "selected_property_id": null
  }
}
```

**Refused messages.** Before any AI call, `/ask` checks the message (English and Bengali). Questions
unrelated to real estate (general knowledge, coding, recipes, sports, poems, translation, …) and jailbreak
attempts ("ignore previous instructions", "show your system prompt", "you are now DAN", …) get a fixed reply
instantly, at no AI cost. The response is `200` with `"blocked": "off_topic"` or `"blocked": "jailbreak"`
and **no** `system_state` (keep the previous one):

```json
{ "answer": "I'm HomeFind AI, a real estate assistant, so I can only help with finding and comparing homes...", "blocked": "off_topic" }
```

Refused messages don't count toward the daily AI caps, but jailbreak attempts count as strikes (see Limits).
Short follow-ups ("is it near a park?", "yes please") are accepted only after the assistant has replied in the chat.

`answer` is Markdown (it may contain tables). Render it with a Markdown library and **sanitize the HTML**
before inserting it into a page.

Conversation stages: `initial_greeting`, `gathering_requirements`, `searching_properties`,
`showing_exact_matches`, `showing_similar_matches`, `providing_details`, `no_matches_found`,
`requesting_contact`, `contact_collected`, `follow_up`. The stage is updated at the start of each turn, so it can lag one message behind the reply.

When the user shares an email or WhatsApp number and both a budget and a location are known, their
requirements are saved to `user_data/user_requirements.csv` (`requirements_saved` becomes `true`).

**Errors:** `400` invalid body, `413` message too long, `429` rate limited, `500` processing failed
(the body still contains a user-friendly `answer`).

---

### POST `/audio/transcribe`

Convert recorded speech to text (OpenAI `gpt-transcribe`, falling back to Groq `whisper-large-v3-turbo`).

**Request:** `multipart/form-data` with one file field, `audio_file`. Supported formats include WebM/Opus,
Ogg, MP4/M4A, MP3 and WAV (max 25 MB).

**Response** `200`

```json
{ "text": "I am looking for a three-bedroom house in Seattle with a budget of $600,000." }
```

**Errors:** `413` too large, `415` not an audio file, `429` rate limited, `502` both providers failed.

```bash
curl -X POST http://localhost:7860/audio/transcribe -F "audio_file=@speech.webm;type=audio/webm"
```

---

### POST `/audio/tts`

Convert text to speech. Uses OpenAI (`gpt-4o-mini-tts`, voice `marin`) and falls back to Groq
(`canopylabs/orpheus-v1-english`, voice `hannah`). Models and voices are set in `.env`
(`OPENAI_TTS_*`, `GROQ_TTS_*`).

**Request body**

```json
{ "text": "I found 3 properties that match your requirements" }
```

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | Yes | Text to speak (truncated to 4,000 chars) |

**Response** `200`

```json
{ "audio": "<base64-encoded audio>", "mime_type": "audio/mpeg" }
```

`mime_type` is `audio/mpeg` from OpenAI and `audio/wav` from Groq. Play it with
`new Audio("data:" + mime_type + ";base64," + audio)`.

**Errors:** `400` missing text, `429` rate limited, `502` both providers failed.

```bash
curl -X POST http://localhost:7860/audio/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how can I help you?"}'
```

---

### GET `/status`

Readiness check. Returns `503` until the vector database and the LLM chain are ready (the first start can
take a few minutes while the property index is built).

**Response** `200`

```json
{
  "status": "ok",
  "database_size": 4090,
  "model": "openai/gpt-oss-120b",
  "providers": { "chat": "groq", "embeddings": "local", "speech": "groq" },
  "retrieval_method": "Vector-based retrieval with FAISS",
  "features": ["Conversational responses", "Semantic search with FAISS", "..."]
}
```

`providers` values: `openai`, `groq`, `local` (embeddings only), or `none` (no working key).

---

### GET `/property_summary`

Statistics for the property dataset (used by the sidebar in the web UI).

**Response** `200`

```json
{
  "total_properties": 4090,
  "avg_price": 559822.13,
  "avg_sqft": 2136.26,
  "price_range": { "min": 80000.0, "max": 26590000.0, "median": 464600.0 }
}
```

---

### GET `/requirements` (admin)

All saved user requirements, including users' contact details.

```bash
curl -u admin:$ADMIN_PASSWORD http://localhost:7860/requirements
```

**Response** `200` (missing values are `null`)

```json
{
  "requirements": [
    {
      "timestamp": "2026-05-22 16:13:03",
      "user_id": "882bda45",
      "email": null,
      "whatsapp": "01305354567",
      "price_min": 300000.0,
      "bedrooms": 2.0,
      "bathrooms": null,
      "sqft_living": null,
      "location": "Seattle",
      "property_type": null,
      "specific_requirements": null
    }
  ]
}
```

**Errors:** `401`, `404` (disabled, or no data saved yet), `500`.

---

### GET / POST `/api-@dmin` (admin)

HTML admin panel. `GET` shows the form and the last 4 characters of the current OpenAI key. `POST` with form
field `api_key` (must look like an OpenAI key, `sk-…`) saves it to `.env` and restarts the app. Open it in a
browser; it will prompt for the admin login.

## Examples

### Python: a two-turn conversation

```python
import json
import requests

BASE = "http://localhost:7860"
history = []

def ask(query):
    reply = requests.post(f"{BASE}/ask", json={"query": query, "chat_history": history}, timeout=60).json()
    history.extend([
        {"role": "user", "content": query},
        {"role": "assistant", "content": reply["answer"]},
        {"role": "system", "content": json.dumps(reply["system_state"])},
    ])
    return reply["answer"]

print(ask("I want a 3 bedroom house in Seattle under 600000"))
print(ask("Tell me more about the first one"))
```

### JavaScript: speak a reply

```javascript
const res = await fetch("/audio/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Here are three homes that match." }),
});
if (res.ok) {
  const { audio, mime_type } = await res.json();
  new Audio(`data:${mime_type};base64,${audio}`).play();
}
```
