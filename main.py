"""
HomeFind AI - Main Application Entry Point
"""
import uvicorn
from app.core.application import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )
