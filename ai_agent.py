import re
import json
from typing import Dict, List, Any, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from data_manager import load_and_preprocess_data, setup_vector_db, save_user_requirements, convert_numpy_types

# Type definitions for our state
class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    retrieved_documents: List[Dict[str, Any]]
    response: str
    user_requirements: Optional[Dict[str, Any]]
    contact_info: Optional[Dict[str, str]]
    conversation_stage: str
    properties_shown: List[int]
    similar_properties: List[Dict[str, Any]]
    needs_contact_info: bool

class RealEstateAgent:
    def __init__(self):
        self.df = None
        self.vector_store = None
        self.llm = None
        self.chain = None
        self.property_index = {}

    def initialize(self):
        self.df, df_for_text, file_mtime = load_and_preprocess_data()
        if self.df is None:
            return False
        # Unpack both returned values here
        _, vector_store = setup_vector_db(self.df, df_for_text, file_mtime)
        if vector_store is None:
            return False
        self.vector_store = vector_store
        self.property_index = {row['property_id']: row.to_dict() for _, row in self.df.iterrows()}
        return True

    def setup_chain(self):
        """Set up the LangChain conversational chain"""
        try:
            print("Setting up language model and chain...")
            
            # Initialize the language model
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo-16k",
                temperature=0.5
            )
            
            # Create the retrieval-augmented generation chain
            vectorstore_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # System template for conversation
            system_template = """You are a professional AI real estate agent assistant named HomeFind AI. 
            Your job is to help users find their ideal home based on their specific requirements.

            CONVERSATION APPROACH:
            1. Begin professionally by introducing yourself as a real estate agent and ask 1-2 specific questions
            2. Gather key requirements one or two at a time (budget, location, bedrooms, etc.)
            3. Continue asking questions until the user indicates they have no more requirements
            4. For exact matches, ALL user requirements must be precisely met
            5. If no exact matches, offer similar properties within ±10% of budget only
            6. Show at most 3 properties at a time with key features and personalized recommendations
            7. Request contact info (WhatsApp/email) when: exact match not found, scheduling a viewing, or conversation ending

            PROPERTY DISPLAY FORMAT:
            - Property ID: [ID]
            - Price: $[PRICE]
            - Bedrooms: [NUMBER]
            - Bathrooms: [NUMBER]
            - Square Footage: [SQFT] sq ft
            - [SPECIAL FEATURES]
            - [PERSONALIZED DESCRIPTION WHY THIS PROPERTY IS SUITABLE]

            DETAILED PROPERTY PRESENTATION:
            When showing full details about a property, include all available information such as:
            - Year built
            - Condition rating
            - View quality
            - Renovation status
            - Lot size
            - Neighborhood features (if available)
            - Property highlights, strengths, and unique selling points

            CONTACT INFO COLLECTION:
            - Always collect WhatsApp/email when showing similar properties
            - Always collect WhatsApp/email when scheduling property viewings
            - Always collect WhatsApp/email when the conversation appears to be ending
            - Save all user requirements to CSV when contact info is provided
            - Confirm saving and explicitly show saved requirements to the user

            CONVERSATION STAGE:
            {conversation_stage}

            USER REQUIREMENTS COLLECTED SO FAR:
            {user_requirements}

            CONTEXTUAL INFORMATION:
            {context}

            Remember to maintain a professional, helpful demeanor throughout the conversation.
            """
            
            # Create prompt templates
            contextual_prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{query}")
            ])
            
            # Define the retrieval function
            def retrieve_documents(state: AgentState) -> AgentState:
                query = state["query"]
                # Use semantic search to find relevant properties
                retrieved_docs = vectorstore_retriever.invoke(query)
                # Update state with retrieved documents
                state["retrieved_documents"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in retrieved_docs
                ]
                return state
            
            # Define the function to extract user requirements
            def extract_requirements(state: AgentState) -> AgentState:
                if not state.get("user_requirements"):
                    state["user_requirements"] = {}
                
                # Extract requirements from user query and previous conversation
                query = state["query"]
                chat_history = state.get("chat_history", [])
                
                # Update requirements based on conversation context
                requirements = state["user_requirements"]
                
                # Process the query for keywords
                budget_match = re.search(r'(?:budget|price|afford|spend|cost).*?(\$?[\d,]+k?|[\d,]+\s*thousand|[\d,]+\s*million)', query, re.IGNORECASE)
                if budget_match:
                    budget_str = budget_match.group(1).replace('$', '').replace(',', '')
                    # Convert k to thousands
                    if 'k' in budget_str.lower():
                        budget_str = budget_str.lower().replace('k', '')
                        budget = float(budget_str) * 1000
                    # Convert thousand/million to numbers
                    elif 'thousand' in budget_str.lower():
                        budget_str = budget_str.lower().replace('thousand', '').strip()
                        budget = float(budget_str) * 1000
                    elif 'million' in budget_str.lower():
                        budget_str = budget_str.lower().replace('million', '').strip()
                        budget = float(budget_str) * 1000000
                    else:
                        budget = float(budget_str)
                    requirements["budget"] = budget
                
                # Look for bedroom requirements
                bedroom_match = re.search(r'(\d+)\s*(?:bed|bedroom|br)', query, re.IGNORECASE)
                if bedroom_match:
                    requirements["bedrooms"] = int(bedroom_match.group(1))
                
                # Look for bathroom requirements
                bathroom_match = re.search(r'(\d+)\s*(?:bath|bathroom|ba)', query, re.IGNORECASE)
                if bathroom_match:
                    requirements["bathrooms"] = int(bathroom_match.group(1))
                
                # Look for square footage requirements
                sqft_match = re.search(r'(\d+)\s*(?:sq\.?\s*ft\.?|square\s*(?:foot|feet)|sqft)', query, re.IGNORECASE)
                if sqft_match:
                    requirements["sqft_living"] = int(sqft_match.group(1))
                
                # Location information
                location_match = re.search(
                    r"""
                    (?:\bin\b|\bat\b|\bnear\b|\baround\b
                    |\blocated\s+in\b|\blooking\s+for\b|\bprefer(?:red)?\b|\barea\b)
                    \s+
                    ([A-Za-z0-9][A-Za-z0-9\s,.\-]+?[A-Za-z0-9])
                    (?=\s*(?:with|and|but|or|for|\.|,|\$|\Z))
                    """, query,
                    re.IGNORECASE | re.VERBOSE
                )
                if location_match:
                    requirements["location"] = location_match.group(1).strip()
                
                # Add property type extraction
                property_type_match = re.search(
                    r'(apartment|condo|house|townhouse|villa|commercial)',
                    query, re.IGNORECASE
                )
                if property_type_match:
                    requirements["property_type"] = property_type_match.group(1).lower()
                    
                # Capture free-form special requirements
                special_req_match = re.search(
                    r'(?:need|require|looking for|want|must have)\s+(.*?)(?:\.|$)',
                    query,
                    re.IGNORECASE
                )
                if special_req_match:
                    requirements["special_requirements"] = special_req_match.group(1).strip()

                
                # Update the state
                state["user_requirements"] = requirements
                print("Extracted requirements so far:", requirements)
                return state
            
            # Define the function to find matching properties
            def find_matching_properties(state: AgentState) -> AgentState:
                if not state.get("user_requirements"):
                    return state
                
                requirements = state["user_requirements"]
                matching_properties = []
                similar_properties = []
                
                # Get filter criteria
                budget = requirements.get("budget")
                bedrooms = requirements.get("bedrooms")
                bathrooms = requirements.get("bathrooms")
                sqft = requirements.get("sqft_living")
                location = requirements.get("location")
                
                # First try exact matching with the database
                filtered_df = self.df.copy()
                
                # Apply exact filters with appropriate tolerances
                if budget:
                    # Use a tighter 5% tolerance for exact match on budget
                    
                    filtered_df = filtered_df[filtered_df['price']== budget]
                
                if bedrooms is not None:
                    filtered_df = filtered_df[filtered_df['bedrooms'] == bedrooms]
                
                if bathrooms is not None:
                    filtered_df = filtered_df[filtered_df['bathrooms'] == bathrooms]
                
                if sqft:
                    # Use a 10% tolerance for exact match on square footage
                    sqft_min = sqft * 0.9
                    sqft_max = sqft * 1.1
                    filtered_df = filtered_df[filtered_df['sqft_living'].between(sqft_min, sqft_max)]
                
                if location:
                    # For location, use simple text matching if available in the dataset
                    if 'city' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['city'].str.lower().str.contains(location.lower(), na=False)]
                    elif 'neighborhood' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['neighborhood'].str.lower().str.contains(location.lower(), na=False)]
                    elif 'zipcode' in filtered_df.columns and location.isdigit():
                        filtered_df = filtered_df[filtered_df['zipcode'] == location]
                
                # Get the matching properties and sort by closest price match
                if not filtered_df.empty:
                    if budget:
                        filtered_df['price_diff'] = abs(filtered_df['price'] - budget)
                        filtered_df = filtered_df.sort_values('price_diff')
                    
                    # Limit to top results and convert to dict
                    matching_properties = filtered_df.head(5).to_dict('records')
                
                # If no exact matches, find similar properties using vector search combined with relaxed filtering
                if len(matching_properties) == 0:
                    # Create a query for semantic search based on requirements
                    semantic_query = "Looking for a property"
                    
                    if budget:
                        semantic_query += f" around ${budget:,.0f}"
                    
                    if bedrooms:
                        semantic_query += f" with {bedrooms} bedrooms"
                    
                    if bathrooms:
                        semantic_query += f" and {bathrooms} bathrooms"
                    
                    if sqft:
                        semantic_query += f" around {sqft} square feet"
                    
                    if location:
                        semantic_query += f" in {location}"
                    
                    # Use vector search to find semantically similar properties
                    similar_docs = self.vector_store.similarity_search(
                        semantic_query, 
                        k=10,  # Get more candidates than needed for filtering
                        filter={"document_type": "property"}  # Only get actual properties, not summaries
                    )
                    
                    # Extract property IDs from the search results
                    property_ids = []
                    for doc in similar_docs:
                        if 'property_id' in doc.metadata:
                            property_ids.append(doc.metadata['property_id'])
                    
                    # Filter the dataframe by these property IDs
                    similar_df = self.df[self.df.index.isin(property_ids)].copy()
                    
                    # Apply relaxed criteria for similar properties
                    if budget:
                        # Use wider ±10% tolerance for similar properties
                        budget_min = budget * 0.9
                        budget_max = budget * 1.1
                        similar_df = similar_df[similar_df['price'].between(budget_min, budget_max)]
                    
                    if bedrooms:
                        # Allow ±1 bedroom for similar properties
                        similar_df = similar_df[(similar_df['bedrooms'] >= bedrooms - 1) & 
                                                (similar_df['bedrooms'] <= bedrooms + 1)]
                    
                    if bathrooms:
                        # Allow ±0.5 bathrooms for similar properties
                        similar_df = similar_df[(similar_df['bathrooms'] >= bathrooms - 0.5) & 
                                                (similar_df['bathrooms'] <= bathrooms + 0.5)]
                    
                    # Sort and get similar properties
                    if not similar_df.empty and budget:
                        similar_df['price_diff'] = abs(similar_df['price'] - budget)
                        similar_df = similar_df.sort_values('price_diff')
                        
                    # Get the top 5 similar properties
                    similar_properties = similar_df.head(5).to_dict('records')
                
                # Update state with matching and similar properties
                state["matching_properties"] = matching_properties
                state["similar_properties"] = similar_properties
                state["needs_contact_info"] = (len(matching_properties) == 0 and len(similar_properties) > 0)
                
                return state
            
            # Define function to decide the next action
            def decide_next_action(state: AgentState) -> AgentState:
                # Initialize conversation stage if not present
                if "conversation_stage" not in state:
                    state["conversation_stage"] = "initial_greeting"
                    return state
                
                current_stage = state["conversation_stage"]
                requirements = state.get("user_requirements", {})
                has_contact_info = "contact_info" in state and (
                    state["contact_info"].get("email") or state["contact_info"].get("whatsapp")
                )
                
                # Look for signals in the user's query that they want to see properties
                query = state["query"].lower()
                wants_to_see_properties = any(phrase in query for phrase in [
                    "show me", "property", "properties", "houses", "listing", 
                    "see what", "available", "options", "show property"
                ])
                
                # Look for signals that conversation is ending
                conversation_ending = any(phrase in query for phrase in [
                    "bye", "goodbye", "thank you", "thanks", "that's all", 
                    "that is all", "end", "finish", "done"
                ])
                
                # Look for signals that the user wants more details
                wants_details = any(phrase in query for phrase in [
                    "tell me more", "more details", "more information", "details", 
                    "features", "about property", "about house", "full details",
                    "specific", "more about"
                ])
                
                # Look for signals about viewing properties
                wants_viewing = any(phrase in query for phrase in [
                    "visit", "see in person", "tour", "viewing", "see it", 
                    "appointment", "schedule", "when can i", "available to see"
                ])
                
                # Force contact collection in specific situations
                if wants_viewing or conversation_ending or not state.get("matching_properties"):
                    state["needs_contact_info"] = True
                
                # Decide the next stage based on current stage and collected information
                if current_stage == "initial_greeting":
                    state["conversation_stage"] = "gathering_requirements"
                
                elif current_stage == "gathering_requirements":
                    if wants_to_see_properties:
                        state["conversation_stage"] = "searching_properties"
                    elif conversation_ending:
                        state["conversation_stage"] = "requesting_contact"
                
                elif current_stage == "searching_properties":
                    if wants_details:
                        state["conversation_stage"] = "providing_details"
                    elif wants_viewing:
                        state["conversation_stage"] = "requesting_contact"
                    elif state.get("needs_contact_info", False) and not has_contact_info:
                        state["conversation_stage"] = "requesting_contact"
                    elif state.get("needs_contact_info", False) and has_contact_info:
                        state["conversation_stage"] = "showing_similar"
                    else:
                        state["conversation_stage"] = "showing_matches"
                
                elif current_stage == "providing_details":
                    if wants_viewing:
                        state["conversation_stage"] = "requesting_contact"
                    elif conversation_ending and not has_contact_info:
                        state["conversation_stage"] = "requesting_contact"
                    else:
                        state["conversation_stage"] = "follow_up"
                
                elif current_stage == "requesting_contact" and has_contact_info:
                    # After collecting contact info, either show similar properties or end
                    if state.get("similar_properties"):
                        state["conversation_stage"] = "showing_similar"
                    else:
                        state["conversation_stage"] = "follow_up"
                
                elif current_stage in ["showing_matches", "showing_similar"]:
                    if wants_details:
                        state["conversation_stage"] = "providing_details"
                    elif wants_viewing:
                        state["conversation_stage"] = "requesting_contact"
                    elif conversation_ending and not has_contact_info:
                        state["conversation_stage"] = "requesting_contact"
                    else:
                        state["conversation_stage"] = "follow_up"
                
                # Ensure contact info is requested appropriately
                if ((not state.get("matching_properties") and state.get("similar_properties")) or
                    wants_viewing or conversation_ending) and not has_contact_info:
                    state["needs_contact_info"] = True
                
                return state
            
            # Define the function to collect contact information
            def collect_contact_info(state: AgentState) -> AgentState:
                query = state["query"].lower()
                
                # Initialize contact info if not present
                if "contact_info" not in state:
                    state["contact_info"] = {}
                
                # Extract email using regex
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                email_match = re.search(email_pattern, query)
                if email_match:
                    state["contact_info"]["email"] = email_match.group(0)
                
                # Extract WhatsApp/phone number using regex
                phone_pattern = r'(?:\+?\d{1,4}[\s\-\.])?(?:\(?\d{2,5}\)?)[\s\-\.]?\d{2,5}[\s\-\.]?\d{2,5}(?:[\s\-\.]?\d{1,5})?(?:\s*(?:ext|x|extension)\s*\d{1,7})?'
                phone_match = re.search(phone_pattern, query)
                if phone_match:
                    state["contact_info"]["whatsapp"] = phone_match.group(0)
                
                return state
            
            # Define the function to generate response
            def generate_response(state: AgentState) -> AgentState:
                # Format the user requirements for the prompt
                user_requirements_str = "No requirements collected yet."
                if state.get("user_requirements"):
                    requirements = state["user_requirements"]
                    user_requirements_str = "Requirements collected:\n"
                    
                    if "budget" in requirements:
                        user_requirements_str += f"- Budget: ${requirements['budget']:,.2f}\n"
                    if "bedrooms" in requirements:
                        user_requirements_str += f"- Bedrooms: {requirements['bedrooms']}\n"
                    if "bathrooms" in requirements:
                        user_requirements_str += f"- Bathrooms: {requirements['bathrooms']}\n"
                    if "sqft_living" in requirements:
                        user_requirements_str += f"- Square Footage: {requirements['sqft_living']} sq ft\n"
                    if "location" in requirements:
                        user_requirements_str += f"- Location: {requirements['location']}\n"
                
                # Format the contextual information based on the conversation stage
                context_info = ""

                # Check if user wants detailed info (could check conversation_stage or query flags)
                detailed_request = state.get("conversation_stage") == "providing_details"
                
                # Add matching properties to context if available (limit to top 3)
                if state.get("matching_properties"):
                    context_info += "\nExact Matching Properties:\n"
                    
                    # Limit to showing 3 properties at a time
                    matches_to_show = state["matching_properties"][:3]
                    
                    for i, prop in enumerate(matches_to_show):
                        # Calculate special features to highlight
                        special_features = []
                        if prop.get('waterfront') == 1:
                            special_features.append("Waterfront property")
                        if prop.get('view', 0) >= 4:
                            special_features.append("Excellent view (rated {}/5)".format(prop.get('view')))
                        if prop.get('condition', 0) >= 4:
                            special_features.append("Great condition (rated {}/5)".format(prop.get('condition')))
                        if prop.get('yr_renovated', 0) > 0:
                            special_features.append(f"Renovated in {prop.get('yr_renovated')}")
                        
                        # Create personalized description based on property features
                        personalized_desc = generate_property_description(prop, state.get("user_requirements", {}), "exact", detailed=detailed_request)
                        
                        context_info += f"\nProperty {i+1} (ID: {prop.get('property_id', '')}):\n"
                        context_info += f"- Price: ${prop['price']:,.2f}\n"
                        context_info += f"- Bedrooms: {prop['bedrooms']}\n"
                        context_info += f"- Bathrooms: {prop['bathrooms']}\n"
                        context_info += f"- Square Footage: {prop['sqft_living']} sq ft\n"
                        if "yr_built" in prop:
                            context_info += f"- Year Built: {prop['yr_built']}\n"
                        
                        # Add special features section
                        if special_features:
                            context_info += f"- Special Features: {', '.join(special_features)}\n"
                        
                        # Add the personalized description
                        context_info += f"{personalized_desc}\n"

                # Similarly, replace the similar properties section:
                if not state.get("matching_properties") and state.get("similar_properties"):
                    context_info += "\nSimilar Properties (within ±10% of budget):\n"
                    
                    # Limit to showing 3 properties at a time
                    similar_to_show = state["similar_properties"][:3]
                    
                    for i, prop in enumerate(similar_to_show):
                        # Calculate why this property is similar but different
                        difference_notes = []
                        requirements = state.get("user_requirements", {})
                        
                        if "budget" in requirements and "price" in prop:
                            price_diff = prop["price"] - requirements["budget"]
                            if price_diff > 0:
                                difference_notes.append(f"${price_diff:,.2f} above budget")
                            else:
                                difference_notes.append(f"${abs(price_diff):,.2f} below budget")
                        
                        if "bedrooms" in requirements and "bedrooms" in prop:
                            bed_diff = prop["bedrooms"] - requirements["bedrooms"]
                            if bed_diff != 0:
                                difference_notes.append(f"{abs(bed_diff)} {'more' if bed_diff > 0 else 'fewer'} bedroom(s)")
                        
                        if "bathrooms" in requirements and "bathrooms" in prop:
                            bath_diff = prop["bathrooms"] - requirements["bathrooms"]
                            if bath_diff != 0:
                                difference_notes.append(f"{abs(bath_diff)} {'more' if bath_diff > 0 else 'fewer'} bathroom(s)")
                        
                        # Format property info
                        context_info += f"\nProperty {i+1} (ID: {prop.get('property_id', '')}):\n"
                        context_info += f"- Price: ${prop['price']:,.2f}\n"
                        context_info += f"- Bedrooms: {prop['bedrooms']}\n"
                        context_info += f"- Bathrooms: {prop['bathrooms']}\n"
                        context_info += f"- Square Footage: {prop['sqft_living']} sq ft\n"
                        if "yr_built" in prop:
                            context_info += f"- Year Built: {prop['yr_built']}\n"
                        
                        # Special features
                        special_features = []
                        if prop.get('waterfront') == 1:
                            special_features.append("Waterfront property")
                        if prop.get('view', 0) >= 4:
                            special_features.append(f"Excellent view (rated {prop.get('view')}/5)")
                        if prop.get('condition', 0) >= 4:
                            special_features.append(f"Great condition (rated {prop.get('condition')}/5)")
                        if prop.get('yr_renovated', 0) > 0:
                            special_features.append(f"Renovated in {prop.get('yr_renovated')}")
                        
                        # Add special features if any
                        if special_features:
                            context_info += f"- Special Features: {', '.join(special_features)}\n"
                        
                        # Add differences and personalized description
                        if difference_notes:
                            context_info += f"- Differences: {', '.join(difference_notes)}\n"
                        
                        # Add property description
                        personalized_desc = generate_property_description(prop, state.get("user_requirements", {}), "similar", detailed=detailed_request)
                        context_info += f"{personalized_desc}\n"
                
                # Add contact info collection status/prompting when needed
                if state.get("needs_contact_info", False):
                    contact_info = state.get("contact_info", {})
                    if not contact_info.get("email") and not contact_info.get("whatsapp"):
                        context_info += "\nAction Needed: Request user's email OR WhatsApp number to save their requirements and notify them of future property matches.\n"
                    elif not contact_info.get("email") or not contact_info.get("whatsapp"):
                        # We have one but not both
                        context_info += "\nAction Needed: Thank the user for providing contact info. Offer to save with just what they've provided or ask for the other contact method for better reachability.\n"
                    else:
                        context_info += "\nContact Information Collected. Offer similar properties within ±10% of budget.\n"
                
                # Add confirmation about saved requirements if applicable
                if state.get("requirements_saved", False):
                    context_info += "\nREQUIREMENTS SAVED CONFIRMATION: User requirements have been saved successfully.\n"
                    context_info += "Saved requirements summary:\n"
                    
                    requirements = state.get("user_requirements", {})
                    if "budget" in requirements:
                        context_info += f"- Budget: ${requirements['budget']:,.2f}\n"
                    if "bedrooms" in requirements:
                        context_info += f"- Bedrooms: {requirements['bedrooms']}\n"
                    if "bathrooms" in requirements:
                        context_info += f"- Bathrooms: {requirements['bathrooms']}\n"
                    if "sqft_living" in requirements:
                        context_info += f"- Square Footage: {requirements['sqft_living']} sq ft\n"
                    if "location" in requirements:
                        context_info += f"- Location: {requirements['location']}\n"
                    
                    contact = state.get("contact_info", {})
                    if contact.get("email"):
                        context_info += f"- Contact Email: {contact['email']}\n"
                    if contact.get("whatsapp"):
                        context_info += f"- Contact WhatsApp: {contact['whatsapp']}\n"

                # Generate response using LLM
                response = self.llm.invoke(
                    contextual_prompt.invoke({
                        "context": context_info,
                        "user_requirements": user_requirements_str,
                        "conversation_stage": state["conversation_stage"],
                        "chat_history": state.get("chat_history", []),
                        "query": state["query"]
                    })
                )
                
                # Update state with response
                state["response"] = response.content
                return state
            
            # Create the LangGraph workflow
            workflow = StateGraph(AgentState)
            
            # Define the nodes
            workflow.add_node("retrieve_documents", retrieve_documents)
            workflow.add_node("extract_requirements", extract_requirements)
            workflow.add_node("find_matching_properties", find_matching_properties)
            workflow.add_node("decide_next_action", decide_next_action)
            workflow.add_node("collect_contact_info", collect_contact_info)
            workflow.add_node("generate_response", generate_response)
            
            # Define the edges
            workflow.add_edge("retrieve_documents", "extract_requirements")
            workflow.add_edge("extract_requirements", "find_matching_properties")
            workflow.add_edge("find_matching_properties", "decide_next_action")
            workflow.add_edge("decide_next_action", "collect_contact_info")
            workflow.add_edge("collect_contact_info", "generate_response")
            
            # Set the entry point
            workflow.set_entry_point("retrieve_documents")
            
            # Compile the graph
            self.chain = workflow.compile()
            
            print("Chain setup complete!")
            return self.chain
            
        except Exception as e:
            print(f"Error setting up chain: {e}")
            return None

def generate_property_description(property_data, requirements, match_type="exact", detailed=False):
    try:
        if detailed:
            property_prompt = f"""
            As an AI real estate agent, provide a detailed property description highlighting:
            - All property features (price, bedrooms, bathrooms, sqft, year built, condition, view, waterfront, renovations)
            - Why this property suits the client's detailed requirements
            - Unique selling points and neighborhood highlights if available
            
            PROPERTY DETAILS:
            - Price: ${property_data.get('price', 'N/A'):,.2f}
            - Bedrooms: {property_data.get('bedrooms', 'N/A')}
            - Bathrooms: {property_data.get('bathrooms', 'N/A')}
            - Square Footage: {property_data.get('sqft_living', 'N/A')} sq ft
            - Year Built: {property_data.get('yr_built', 'N/A')}
            - Property Condition: {property_data.get('condition', 'N/A')}/5
            - View Quality: {property_data.get('view', 'N/A')}/5
            - Waterfront: {"Yes" if property_data.get('waterfront', 0) == 1 else "No"}
            - Year Renovated: {property_data.get('yr_renovated', 0) if property_data.get('yr_renovated', 0) > 0 else "Not renovated"}

            CLIENT REQUIREMENTS:
            - Budget: ${requirements.get('budget', 'Not specified'):,.2f}
            - Desired Bedrooms: {requirements.get('bedrooms', 'Not specified')}
            - Desired Bathrooms: {requirements.get('bathrooms', 'Not specified')}
            - Desired Square Footage: {requirements.get('sqft_living', 'Not specified')} sq ft
            - Desired Location: {requirements.get('location', 'Not specified')}
            - Property Type: {requirements.get('property_type', 'Not specified')}
            - Special Requirements: {requirements.get('special_requirements', 'None')}

            MATCH TYPE: {match_type.upper()} MATCH

            Your description should be thorough and highlight all relevant details to help the client make an informed decision.

            PROPERTY DESCRIPTION:
            """
        else:
            property_prompt = f"""
            As an AI real estate agent, craft a brief but compelling personalized property description (1-2 sentences) 
            explaining why this property matches the client's needs.

            PROPERTY DETAILS:
            - Price: ${property_data.get('price', 'N/A'):,.2f}
            - Bedrooms: {property_data.get('bedrooms', 'N/A')}
            - Bathrooms: {property_data.get('bathrooms', 'N/A')}
            - Square Footage: {property_data.get('sqft_living', 'N/A')} sq ft
            - Year Built: {property_data.get('yr_built', 'N/A')}
            - Property Condition: {property_data.get('condition', 'N/A')}/5
            - View Quality: {property_data.get('view', 'N/A')}/5
            - Waterfront: {"Yes" if property_data.get('waterfront', 0) == 1 else "No"}
            - Year Renovated: {property_data.get('yr_renovated', 0) if property_data.get('yr_renovated', 0) > 0 else "Not renovated"}

            CLIENT REQUIREMENTS:
            - Budget: ${requirements.get('budget', 'Not specified'):,.2f}
            - Desired Bedrooms: {requirements.get('bedrooms', 'Not specified')}
            - Desired Bathrooms: {requirements.get('bathrooms', 'Not specified')}
            - Desired Square Footage: {requirements.get('sqft_living', 'Not specified')} sq ft
            - Desired Location: {requirements.get('location', 'Not specified')}
            - Property Type: {requirements.get('property_type', 'Not specified')}
            - Special Requirements: {requirements.get('special_requirements', 'None')}

            MATCH TYPE: {match_type.upper()} MATCH

            Your description should:
            1. Be personalized to the client's specific needs
            2. Highlight 1-2 features that perfectly align with requirements
            3. For "similar" matches, acknowledge the differences but emphasize the value
            4. Include unique selling points (waterfront, view, recent renovation)
            5. Be concise but compelling (1-2 sentences only)

            PROPERTY DESCRIPTION:
            """

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.8
        )
        response = llm.invoke(property_prompt)
        description = response.content.strip()

        if not description:
            if match_type == "exact":
                return "This property is an excellent match for your requirements, meeting all your criteria perfectly."
            else:
                return "This property is similar to what you're looking for and offers great value within your budget range."

        return description

    except Exception as e:
        print(f"Error generating property description: {e}")
        if match_type == "exact":
            return "This property is an exact match for your requirements."
        else:
            return "This property is similar to what you're looking for and may be worth considering."

def process_query(agent, query: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Process a user query and generate a response"""
    
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
        "similar_properties": [],
        "needs_contact_info": False,
        "requirements_saved": False  # Add this flag to track if requirements were saved
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
                    "similar_properties": system_state.get("similar_properties", []),
                    "needs_contact_info": system_state.get("needs_contact_info", False),
                    "requirements_saved": system_state.get("requirements_saved", False)
                })
                break
            except json.JSONDecodeError:
                continue

    # Preserve the actual chat history (user + assistant messages)
    state["chat_history"] = [msg for msg in chat_history if msg["role"] in ["user", "assistant"]]

    try:
        # Process with LangGraph workflow
        result = agent.chain.invoke(state)
        
        # Format the response with full system state
        response = {
            "answer": result["response"],
            "system_state": {
                "type": "system",
                "requirements": result.get("user_requirements", {}),
                "conversation_stage": result.get("conversation_stage", "gathering_requirements"),
                "contact_info": result.get("contact_info", {}),
                "properties_shown": result.get("properties_shown", []),
                "similar_properties": result.get("similar_properties", []),
                "needs_contact_info": result.get("needs_contact_info", False),
                "requirements_saved": result.get("requirements_saved", False)
            }
        }

        # Save requirements only when contact info is provided and not already saved
        if (response["system_state"]["contact_info"].get("email") or 
            response["system_state"]["contact_info"].get("whatsapp")) and not response["system_state"]["requirements_saved"]:
            save_success = save_user_requirements({
                "user_requirements": response["system_state"]["requirements"],
                "contact_info": response["system_state"]["contact_info"],
                "conversation_stage": response["system_state"]["conversation_stage"]
            })
            response["system_state"]["requirements_saved"] = save_success

        # Add matching properties if available
        if "matching_properties" in result and result["matching_properties"]:
            response["matching_properties"] = result["matching_properties"][:5]

        # Add similar properties if available
        if "similar_properties" in result and result["similar_properties"]:
            response["similar_properties"] = result["similar_properties"][:5]

        return response

    except Exception as e:
        print(f"Error processing query: {e}")
        return {"answer": "I'm sorry, but I encountered an error while processing your request. Please try again."}