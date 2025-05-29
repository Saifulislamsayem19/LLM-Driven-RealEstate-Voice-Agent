import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
import pandas as pd

from ai_agent import RealEstateAgent, process_query, convert_numpy_types
from data_manager import USER_REQUIREMENTS_FILE
from audio import audio_bp

load_dotenv()

app = Flask(__name__)
app.register_blueprint(audio_bp, url_prefix="/audio")

# Initialize the agent instance globally
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    chat_history = data.get("chat_history", [])

    if not query:
        return jsonify({"answer": "Welcome to our real estate assistant! How can I help you today?"})

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

        return jsonify(response_data)

    except Exception as e:
        print(f"/ask error: {e}")
        return jsonify({
            "answer": "I'm having trouble processing your request. Please try again later.",
            "system_state": {"type": "system", "error": str(e)}
        }), 500

@app.route("/requirements", methods=["GET"])
def view_requirements():
    """Admin endpoint to view saved user requirements"""
    if not os.path.exists(USER_REQUIREMENTS_FILE):
        return jsonify({"error": "No requirements data available"})
    try:
        df_requirements = pd.read_csv(USER_REQUIREMENTS_FILE)
        return jsonify({"requirements": df_requirements.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": f"Error reading requirements data: {str(e)}"})

@app.route("/property_summary", methods=["GET"])
def property_summary():
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
        return jsonify(summary)
    else:
        return jsonify({"error": "Data not available"})

@app.route("/status", methods=["GET"])
def status():
    if agent.df is None or agent.vector_store is None or agent.chain is None:
        return jsonify({
            "status": "error",
            "message": "System not fully initialized"
        })

    return jsonify({
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
    })

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables or .env file")
        print("Please make sure you have set the OPENAI_API_KEY in your .env file")
        exit(1)

    print("Initializing the real estate chatbot with vector database...")
    if not init_chatbot():
        print("Failed to initialize chatbot. Exiting.")
        exit(1)

    print("Starting the Flask server...")
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=False,  # Prevent socket issues
        threaded=True        # Better support on Windows
    )
