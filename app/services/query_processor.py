"""
process_query — thin service layer that drives the LangGraph agent.

Extracted from the original ai_agent.py process_query function,
logic unchanged.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from app.services.agent import RealEstateAgent
from app.utils.data_manager import convert_numpy_types, save_user_requirements


def process_query(
    agent: RealEstateAgent,
    query: str,
    chat_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Process a user query through the LangGraph workflow and return a structured response."""

   # Initialize state with default values
    state = {
        "query": query,
        "chat_history": [],
        "retrieved_documents": [],
        "response": "",
        "user_requirements": {}, 
        "contact_info": {},
        "conversation_stage": "initial_greeting",
        "properties_shown": [],
        "exact_matches": [],
        "similar_matches": [],
        "needs_contact_info": False,
        "no_properties_found": False,
        "requirements_saved": False,
        "selected_property_id": None
    }

    # Extract previous state from chat history
    for msg in reversed(chat_history):
        if msg.get("role") == "system" and "content" in msg:
            try:
                system_state = json.loads(msg["content"])
                state.update({
                    "user_requirements": system_state.get("requirements", {}),
                    "contact_info": system_state.get("contact_info", {}),
                    "conversation_stage": system_state.get("conversation_stage", "initial_greeting"),
                    "properties_shown": system_state.get("properties_shown", []),
                    "exact_matches": system_state.get("exact_matches", []),
                    "similar_matches": system_state.get("similar_matches", []),
                    "needs_contact_info": system_state.get("needs_contact_info", False),
                    "no_properties_found": system_state.get("no_properties_found", False),
                    "requirements_saved": system_state.get("requirements_saved", False),
                    "selected_property_id": system_state.get("selected_property_id", None)
                })
                break
            except json.JSONDecodeError:
                continue

    # Preserve the actual chat history (user + assistant messages)
    state["chat_history"] = [msg for msg in chat_history if msg["role"] in ["user", "assistant"]]

    try:
        # Process with enhanced LangGraph workflow
        result = agent.chain.invoke(state)
        
        # Format the response with comprehensive system state
        response = {
            "answer": result["response"],
            "system_state": {
                "type": "system",
                "requirements": result.get("user_requirements", {}),
                "conversation_stage": result.get("conversation_stage", "gathering_requirements"),
                "contact_info": result.get("contact_info", {}),
                "properties_shown": result.get("properties_shown", []),
                "exact_matches": result.get("exact_matches", []),
                "similar_matches": result.get("similar_matches", []),
                "needs_contact_info": result.get("needs_contact_info", False),
                "no_properties_found": result.get("no_properties_found", False),
                "requirements_saved": result.get("requirements_saved", False),
                "selected_property_id": result.get("selected_property_id", None)
            }
        }

        # Enhanced requirements saving with validation
        contact_info = response["system_state"]["contact_info"]
        requirements = response["system_state"]["requirements"]
        
        if ((contact_info.get("email") or contact_info.get("whatsapp")) and 
            requirements and not response["system_state"]["requirements_saved"]):
            
            # Validate  meaningful requirements
            has_budget = requirements.get("budget") or requirements.get("budget_min") or requirements.get("budget_max")
            has_location = requirements.get("location")
            
            if has_budget and has_location:
                # Prepare data for saving
                save_data = {
                    "user_requirements": requirements,
                    "contact_info": contact_info,
                    "conversation_stage": response["system_state"]["conversation_stage"]
                }
                
                # Convert numpy types before saving
                save_data = convert_numpy_types(save_data)
                
                save_success = save_user_requirements(save_data)
                response["system_state"]["requirements_saved"] = save_success
                
                if save_success:
                    print("Enhanced user requirements saved successfully to CSV")
                    print(f"Saved requirements: {requirements}")

        # Add property data to response for reference (dataset only)
        if "exact_matches" in result and result["exact_matches"]:
            response["exact_matches"] = result["exact_matches"][:3]  

        if "similar_matches" in result and result["similar_matches"]:
            response["similar_matches"] = result["similar_matches"][:3] 

        # Add validation info
        response["validation_info"] = {
            "has_budget": bool(requirements.get("budget") or requirements.get("budget_min") or requirements.get("budget_max")) if requirements else False,
            "has_location": bool(requirements.get("location")) if requirements else False,
            "properties_from_dataset": True,  
            "comprehensive_filtering_active": True
        }

        return response

    except Exception as e:
        print(f"Error processing query with enhanced filtering: {e}")
        return {
            "answer": "I apologize, but I encountered an issues while searching our property database with your comprehensive criteria. Please try rephrasing your request or contact our support team.",
            "system_state": {
                "type": "system",
                "requirements": state.get("user_requirements", {}),
                "conversation_stage": "error",
                "contact_info": state.get("contact_info", {}),
                "properties_shown": [],
                "exact_matches": [],
                "similar_matches": [],
                "needs_contact_info": False,
                "no_properties_found": True,
                "requirements_saved": False,
                "selected_property_id": None
            },
            "validation_info": {
                "has_budget": False,
                "has_location": False,
                "properties_from_dataset": True,
                "comprehensive_filtering_active": True,
                "error_occurred": True
            }
        }
