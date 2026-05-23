"""
Chat endpoints — /ask and / (homepage).
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.services.agent import agent
from app.utils.data_manager import convert_numpy_types
from app.services.query_processor import process_query

router = APIRouter(tags=["chat"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return request.app.state.templates.TemplateResponse("index.html", {"request": request})


@router.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query: str = data.get("query", "")
    chat_history: list = data.get("chat_history", [])

    if not query:
        return JSONResponse(content={"answer": "Welcome to our real estate assistant! How can I help you today?"})

    try:
        result = process_query(agent, query, chat_history)
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
        print(f"/ask error: {exc}")
        return JSONResponse(
            content={
                "answer": "I'm having trouble processing your request. Please try again later.",
                "system_state": {"type": "system", "error": str(exc)},
            },
            status_code=500,
        )
