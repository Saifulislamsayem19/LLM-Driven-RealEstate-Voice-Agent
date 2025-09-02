import os
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
from audio import audio_router
from ai_agent import RealEstateAgent, process_query, convert_numpy_types
from data_manager import USER_REQUIREMENTS_FILE

load_dotenv()

# Initialize the agent instance
agent = RealEstateAgent()

def init_chatbot():
    """Initialize the agent by loading data, setting up vector db and chain"""
    if not agent.initialize():
        print("Failed to initialize data or vector database.")
        return False
    if not agent.setup_chain():
        print("Failed to setup LangChain workflow.")
        return False
    print("Agent initialization complete.")
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources before the app starts and clean up after it shuts down."""
    # Initialize chatbot
    if not init_chatbot():
        raise RuntimeError("Failed to initialize the chatbot")
    yield
    print("Shutting down...")

# Initialize FastAPI app
app = FastAPI(title="Real Estate Assistant API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include audio router
app.include_router(audio_router)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query", "")
    chat_history = data.get("chat_history", [])

    if not query:
        return JSONResponse(content={"answer": "Welcome to our real estate assistant! How can I help you today?"})

    try:
        result = process_query(agent, query, chat_history)
        converted_result = convert_numpy_types(result)

        system_state = converted_result.get("system_state", {})
        response_data = {
            "answer": converted_result["answer"],
            "system_state": system_state
        }

        if "matching_properties" in converted_result:
            response_data["matching_properties"] = converted_result["matching_properties"]
        if "similar_properties" in converted_result:
            response_data["similar_properties"] = converted_result["similar_properties"]

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"/ask error: {e}")
        return JSONResponse(
            content={
                "answer": "I'm having trouble processing your request. Please try again later.",
                "system_state": {"type": "system", "error": str(e)}
            },
            status_code=500
        )

@app.get("/requirements")
async def view_requirements():
    """Admin endpoint to view saved user requirements"""
    if not os.path.exists(USER_REQUIREMENTS_FILE):
        raise HTTPException(status_code=404, detail="No requirements data available")
    try:
        df_requirements = pd.read_csv(USER_REQUIREMENTS_FILE)
        return {"requirements": df_requirements.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading requirements data: {str(e)}")

@app.get("/property_summary")
async def property_summary():
    if agent.df is not None:
        summary = {
            "total_properties": len(agent.df),
            "avg_price": float(agent.df["price"].mean()) if "price" in agent.df.columns and not agent.df["price"].isnull().all() else None,
            "avg_sqft": float(agent.df["sqft_living"].mean()) if "sqft_living" in agent.df.columns and not agent.df["sqft_living"].isnull().all() else None,
            "price_range": {
                "min": float(agent.df["price"].min()) if "price" in agent.df.columns and not agent.df["price"].isnull().all() else None,
                "max": float(agent.df["price"].max()) if "price" in agent.df.columns and not agent.df["price"].isnull().all() else None,
                "median": float(agent.df["price"].median()) if "price" in agent.df.columns and not agent.df["price"].isnull().all() else None
            }
        }
        return summary
    else:
        raise HTTPException(status_code=500, detail="Data not available")

@app.get("/status")
async def status():
    if agent.df is None or agent.vector_store is None or agent.chain is None:
        raise HTTPException(status_code=503, detail="System initializing")

    return {
        "status": "ok",
        "database_size": len(agent.df),
        "model": "OpenAI GPT-3.5 Turbo",
        "retrieval_method": "Vector-based retrieval with FAISS",
        "features": [
            "Conversational responses",
            "Semantic search with FAISS",
            "Query analysis and transformation",
            "Property-specific metadata filtering",
            "User requirements collection",
            "Contact information storage",
            "Property matching notifications"
        ]
    }

@app.on_event("startup")
async def startup_event():
    if not init_chatbot():
        raise RuntimeError("Failed to initialize the chatbot")
    
if __name__ == "__main__":
   
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )
