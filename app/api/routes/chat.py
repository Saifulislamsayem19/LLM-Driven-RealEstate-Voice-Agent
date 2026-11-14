"""
Chat endpoints — /ask and / (homepage).
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import abuse_guard, chat_rate_limit
from app.services.agent import agent
from app.utils.data_manager import convert_numpy_types
from app.services.query_processor import process_query

router = APIRouter(tags=["chat"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return request.app.state.templates.TemplateResponse(request, "index.html", {"request": request})


@router.post("/ask", dependencies=[Depends(chat_rate_limit)])
async def ask(request: Request):
    try:
        data = await request.json()
    except ValueError:
        return JSONResponse(content={"answer": "Invalid request."}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse(content={"answer": "Invalid request."}, status_code=400)

    query = data.get("query", "")
    chat_history = data.get("chat_history", [])
    if not isinstance(query, str) or not isinstance(chat_history, list):
        return JSONResponse(content={"answer": "Invalid request."}, status_code=400)
    if len(query) > settings.MAX_QUERY_CHARS:
        return JSONResponse(
            content={"answer": f"Your message is too long. Please keep it under {settings.MAX_QUERY_CHARS} characters."},
            status_code=413,
        )
    # Keep only well-formed recent messages; older turns are not needed and only add cost
    chat_history = [
        {"role": m["role"], "content": m["content"][: settings.MAX_QUERY_CHARS * 5]}
        for m in chat_history[-settings.MAX_HISTORY_MESSAGES:]
        if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system") and isinstance(m.get("content"), str)
    ]

    if not query:
        return JSONResponse(content={"answer": "Welcome to our real estate assistant! How can I help you today?"})

    try:
        # LLM calls block; run them off the event loop so other requests aren't stalled
        result = await run_in_threadpool(process_query, agent, query, chat_history)
        converted = convert_numpy_types(result)

        response_data = {
            "answer": converted["answer"],
            "system_state": converted.get("system_state", {}),
        }
        if "matching_properties" in converted:
            response_data["matching_properties"] = converted["matching_properties"]
        if "similar_properties" in converted:
            response_data["similar_properties"] = converted["similar_properties"]

        return JSONResponse(content=response_data)

    except Exception as exc:
        print(f"/ask error: {exc!r}")
        return JSONResponse(
            content={
                "answer": "I'm having trouble processing your request. Please try again later.",
                "system_state": {"type": "system", "error": "internal_error"},
            },
            status_code=500,
        )
