import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
import sys
import threading
import time
import signal
from audio import audio_router
from ai_agent import RealEstateAgent, process_query, convert_numpy_types
from data_manager import USER_REQUIREMENTS_FILE

load_dotenv()

# Global variable to control restart
RESTART_REQUESTED = False

def restart_application():
    """Restart the application completely"""
    print("🔄 Restarting application with new API key...")
    time.sleep(2)  # Give time for response to be sent
    python = sys.executable
    os.execl(python, python, *sys.argv)

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
    # Clear restart flag on startup
    clear_restart_flag()
    
    # Initialize chatbot
    if not init_chatbot():
        raise RuntimeError("Failed to initialize the chatbot")
    
    yield
    
    print("Shutting down...")

def clear_restart_flag():
    """Clear the restart flag from .env file"""
    env_file_path = Path('.env')
    if env_file_path.exists():
        try:
            with open(env_file_path, 'r') as env_file:
                lines = env_file.readlines()
            
            # Remove any restart flags
            with open(env_file_path, 'w') as env_file:
                for line in lines:
                    if not line.strip().startswith('APP_RESTART_PENDING='):
                        env_file.write(line)
        except Exception as e:
            print(f"Warning: Could not clear restart flag: {e}")

def set_restart_flag():
    """Set restart flag in .env file"""
    env_file_path = Path('.env')
    env_vars = {}
    
    # Read existing environment variables
    if env_file_path.exists():
        with open(env_file_path, 'r') as env_file:
            for line in env_file:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    
    # Set restart flag
    env_vars['APP_RESTART_PENDING'] = 'True'
    
    # Write back to .env file
    with open(env_file_path, 'w') as env_file:
        for key, value in env_vars.items():
            env_file.write(f"{key}={value}\n")

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

@app.api_route("/api-@dmin", methods=["GET", "POST"])
async def admin_panel(
    request: Request,
    success: bool = Query(False),
    restarting: bool = Query(False),
    api_key: str = Form(None)
):
    """Admin panel for API key configuration - handles both GET and POST"""
    
    if request.method == "POST":
        # Handle form submission
        try:
            if api_key:
                # Update the .env file
                env_file_path = Path('.env')
                env_vars = {}
                
                # Read existing environment variables
                if env_file_path.exists():
                    with open(env_file_path, 'r') as env_file:
                        for line in env_file:
                            if '=' in line and not line.startswith('#'):
                                key, value = line.strip().split('=', 1)
                                env_vars[key] = value
                
                # Update OpenAI API key and set restart flag
                old_key = env_vars.get('OPENAI_API_KEY', '')
                env_vars['OPENAI_API_KEY'] = api_key
                env_vars['APP_RESTART_PENDING'] = 'True'
                
                # Write back to .env file
                with open(env_file_path, 'w') as env_file:
                    for key, value in env_vars.items():
                        env_file.write(f"{key}={value}\n")
                
                # Update current environment
                os.environ['OPENAI_API_KEY'] = api_key
                
                print(f"✅ OpenAI API key updated from {old_key[:10]}... to {api_key[:10]}...")
                
                # Start restart in background thread
                restart_thread = threading.Thread(target=restart_application)
                restart_thread.daemon = True
                restart_thread.start()
                
                # Show restarting message
                return templates.TemplateResponse(
                    "admin.html", 
                    {
                        "request": request, 
                        "success": True,
                        "restarting": True,
                        "current_api_key": api_key
                    }
                )
            else:
                raise ValueError("API key cannot be empty")
            
        except Exception as e:
            print(f"❌ Error updating API key: {e}")
            current_api_key = os.getenv("OPENAI_API_KEY", "")
            return templates.TemplateResponse(
                "admin.html", 
                {
                    "request": request, 
                    "success": False,
                    "error": str(e),
                    "current_api_key": current_api_key
                }
            )
    
    # Handle GET request - show the form
    current_api_key = os.getenv("OPENAI_API_KEY", "")
    
    # Check if we just restarted
    just_restarted = False
    if not restarting:
        env_file_path = Path('.env')
        if env_file_path.exists():
            with open(env_file_path, 'r') as env_file:
                for line in env_file:
                    if line.startswith('APP_RESTART_PENDING='):
                        just_restarted = True
                        break
    
    return templates.TemplateResponse(
        "admin.html", 
        {
            "request": request, 
            "success": success or just_restarted,
            "restarting": restarting,
            "current_api_key": current_api_key
        }
    )

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

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )