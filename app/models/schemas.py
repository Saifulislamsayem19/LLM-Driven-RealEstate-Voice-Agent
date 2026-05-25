"""
Pydantic schemas for API request and response payloads.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, EmailStr, Field


# Type definitions for the Agent state
class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    retrieved_documents: List[Dict[str, Any]]
    response: str
    user_requirements: Optional[Dict[str, Any]]
    contact_info: Optional[Dict[str, str]]
    conversation_stage: str
    properties_shown: List[int]
    exact_matches: List[Dict[str, Any]]
    similar_matches: List[Dict[str, Any]]
    needs_contact_info: bool
    no_properties_found: bool
    selected_property_id: Optional[int]  