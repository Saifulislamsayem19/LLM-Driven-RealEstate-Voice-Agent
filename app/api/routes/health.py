"""
Health-check and property-summary endpoints.
"""
from fastapi import APIRouter, HTTPException, Request

from app.services.agent import agent
from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/status")
async def status():
    if agent.df is None or agent.vector_store is None or agent.chain is None:
        raise HTTPException(status_code=503, detail="System initialising")

    return {
        "status": "ok",
        "database_size": len(agent.df),
        "model": settings.OPENAI_CHAT_MODEL,
        "retrieval_method": "Vector-based retrieval with FAISS",
        "features": [
            "Conversational responses",
            "Semantic search with FAISS",
            "Query analysis and transformation",
            "Property-specific metadata filtering",
            "User requirements collection",
            "Contact information storage",
            "Property matching notifications",
        ],
    }


@router.get("/property_summary")
async def property_summary():
    if agent.df is None:
        raise HTTPException(status_code=500, detail="Data not available")

    df = agent.df
    price_col_ok = "price" in df.columns and not df["price"].isnull().all()
    sqft_col_ok = "sqft_living" in df.columns and not df["sqft_living"].isnull().all()

    return {
        "total_properties": len(df),
        "avg_price": float(df["price"].mean()) if price_col_ok else None,
        "avg_sqft": float(df["sqft_living"].mean()) if sqft_col_ok else None,
        "price_range": {
            "min": float(df["price"].min()) if price_col_ok else None,
            "max": float(df["price"].max()) if price_col_ok else None,
            "median": float(df["price"].median()) if price_col_ok else None,
        },
    }
