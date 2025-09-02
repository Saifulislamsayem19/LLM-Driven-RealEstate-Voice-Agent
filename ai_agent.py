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
    exact_matches: List[Dict[str, Any]]
    similar_matches: List[Dict[str, Any]]
    needs_contact_info: bool
    no_properties_found: bool
    selected_property_id: Optional[int]  # New field for property details

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

    def generate_property_description(self, prop: Dict[str, Any]) -> str:
        """Generate a personalized description for a property"""
        descriptions = []
        
        # Basic appeal
        bedrooms = prop.get('bedrooms', 0)
        bathrooms = prop.get('bathrooms', 0)
        
        if bedrooms >= 4:
            descriptions.append("Spacious family home")
        elif bedrooms == 3:
            descriptions.append("Perfect for growing families")
        elif bedrooms == 2:
            descriptions.append("Ideal for couples or small families")
        else:
            descriptions.append("Cozy living space")
        
        # Location appeal
        city = prop.get('city', '')
        if city:
            descriptions.append(f"located in the desirable {city} area")
        
        # Special features
        waterfront = prop.get('waterfront', 0)
        view = prop.get('view', 0)
        condition = prop.get('condition', 0)
        floors = prop.get('floors', 0)
        
        if waterfront == 1:
            descriptions.append("with stunning waterfront views")
        elif view >= 4:
            descriptions.append("boasting excellent scenic views")
        elif view >= 3:
            descriptions.append("with pleasant views")
        
        if condition >= 4:
            descriptions.append("in excellent condition")
        elif condition >= 3:
            descriptions.append("well-maintained")
        
        if floors >= 2:
            descriptions.append(f"featuring {floors} levels of living space")
        
        # Year built appeal
        yr_built = prop.get('yr_built', 0)
        yr_renovated = prop.get('yr_renovated', 0)
        current_year = 2024
        
        if yr_renovated and yr_renovated > current_year - 10:
            descriptions.append(f"recently renovated in {yr_renovated}")
        elif yr_built and yr_built > current_year - 10:
            descriptions.append("featuring modern construction")
        elif yr_built and yr_built > current_year - 30:
            descriptions.append("with established charm")
        
        # Square footage appeal
        sqft_living = prop.get('sqft_living', 0)
        if sqft_living > 3000:
            descriptions.append("offering generous living space")
        elif sqft_living > 2000:
            descriptions.append("with ample room to spread out")
        
        # Basement
        sqft_basement = prop.get('sqft_basement', 0)
        if sqft_basement > 500:
            descriptions.append("including finished basement space")
        
        # Combine descriptions
        if descriptions:
            return ". ".join(descriptions) + "."
        else:
            return "A wonderful property that could be perfect for you."

    def format_property_brief(self, prop: Dict[str, Any], index: int) -> str:
        """Format property for brief listing with description"""
        property_id = prop.get('property_id', 'Unknown')
        formatted = f"\n🏠 **Property {index}** (ID: {property_id})\n"
        formatted += f"💰 **Price:** ${prop.get('price', 0):,.0f}\n"
        formatted += f"🛏️ **Bedrooms:** {prop.get('bedrooms', 'N/A')}\n"
        formatted += f"🛁 **Bathrooms:** {prop.get('bathrooms', 'N/A')}\n"
        formatted += f"📐 **Square Footage:** {prop.get('sqft_living', 'N/A')} sq ft\n"
        formatted += f"📍 **Location:** {prop.get('city', 'N/A')}, {prop.get('statezip', 'N/A')}\n"
        
        # Add description
        description = self.generate_property_description(prop)
        formatted += f"✨ **Description:** {description}\n"
        
        # Add one standout feature
        if prop.get('waterfront') == 1:
            formatted += f"🌊 **Highlight:** Waterfront Property\n"
        elif prop.get('view', 0) >= 4:
            formatted += f"🏔️ **Highlight:** Excellent Views (Rating: {prop.get('view')}/5)\n"
        elif prop.get('condition', 0) >= 4:
            formatted += f"✅ **Highlight:** Excellent Condition (Rating: {prop.get('condition')}/5)\n"
        elif prop.get('yr_renovated', 0) > 2010:
            formatted += f"🔨 **Highlight:** Recently Renovated ({prop.get('yr_renovated')})\n"
        
        
        return formatted

    def format_property_full_details(self, prop: Dict[str, Any]) -> str:
        """Format complete property details"""
        property_id = prop.get('property_id', 'Unknown')
        formatted = f"\n🏠 **COMPLETE PROPERTY DETAILS - ID: {property_id}**\n"
        formatted += "=" * 50 + "\n"
        
        # Basic Information
        formatted += f"💰 **Price:** ${prop.get('price', 0):,.0f}\n"
        formatted += f"🛏️ **Bedrooms:** {prop.get('bedrooms', 'N/A')}\n"
        formatted += f"🛁 **Bathrooms:** {prop.get('bathrooms', 'N/A')}\n"
        formatted += f"📐 **Living Space:** {prop.get('sqft_living', 'N/A')} sq ft\n"
        formatted += f"🏡 **Lot Size:** {prop.get('sqft_lot', 'N/A')} sq ft\n"
        formatted += f"🏢 **Floors:** {prop.get('floors', 'N/A')}\n"
        
        # Location Details
        formatted += f"\n📍 **LOCATION INFORMATION:**\n"
        formatted += f"   🏘️ Street: {prop.get('street', 'N/A')}\n"
        formatted += f"   🌆 City: {prop.get('city', 'N/A')}\n"
        formatted += f"   📮 State/Zip: {prop.get('statezip', 'N/A')}\n"
        formatted += f"   🌍 Country: {prop.get('country', 'N/A')}\n"
        
        # Property Features
        formatted += f"\n🏠 **PROPERTY FEATURES:**\n"
        formatted += f"   🌊 Waterfront: {'Yes' if prop.get('waterfront') == 1 else 'No'}\n"
        formatted += f"   🏔️ View Rating: {prop.get('view', 'N/A')}/5\n"
        formatted += f"   ✅ Condition Rating: {prop.get('condition', 'N/A')}/5\n"
        formatted += f"   ⬆️ Above Ground: {prop.get('sqft_above', 'N/A')} sq ft\n"
        formatted += f"   ⬇️ Basement: {prop.get('sqft_basement', 'N/A')} sq ft\n"
        
        # Construction Details
        formatted += f"\n🔨 **CONSTRUCTION DETAILS:**\n"
        formatted += f"   📅 Year Built: {prop.get('yr_built', 'N/A')}\n"
        yr_renovated = prop.get('yr_renovated', 0)
        formatted += f"   🔄 Year Renovated: {yr_renovated if yr_renovated > 0 else 'Not Renovated'}\n"
        
        # Property Age Analysis
        if prop.get('yr_built'):
            age = 2024 - prop.get('yr_built')
            formatted += f"   📊 Property Age: {age} years\n"
        
        # Special Highlights
        highlights = []
        if prop.get('waterfront') == 1:
            highlights.append("🌊 Waterfront Property")
        if prop.get('view', 0) >= 4:
            highlights.append(f"🏔️ Excellent Views ({prop.get('view')}/5)")
        if prop.get('condition', 0) >= 4:
            highlights.append(f"✅ Excellent Condition ({prop.get('condition')}/5)")
        if prop.get('yr_renovated', 0) > 2010:
            highlights.append(f"🔨 Recently Renovated ({prop.get('yr_renovated')})")
        if prop.get('sqft_living', 0) > 3000:
            highlights.append("🏠 Spacious Living Area")
        if prop.get('floors', 0) >= 2:
            highlights.append(f"🏢 Multi-Level Home ({prop.get('floors')} floors)")
        
        if highlights:
            formatted += f"\n⭐ **SPECIAL HIGHLIGHTS:**\n"
            for highlight in highlights:
                formatted += f"   • {highlight}\n"
        
        # Personalized Description
        description = self.generate_property_description(prop)
        formatted += f"\n✨ **PROPERTY OVERVIEW:**\n{description}\n"
        
        formatted += "\n📞 **Interested in this property? Let me know and I can help arrange a viewing!**\n"
        formatted += "=" * 50 + "\n"
        
        return formatted

    def setup_chain(self):
        """Set up the LangChain conversational chain"""
        try:
            print("Setting up language model and chain...")
            
            # Initialize the language model
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo-16k",
                temperature=0.3  # Lower temperature for more consistent requirement extraction
            )
            
            # System template for conversation
            system_template = """You are a professional AI real estate agent assistant named HomeFind AI. 
            Your job is to help users find their ideal home based on their specific requirements.

            CRITICAL RULES:
            1. ONLY suggest properties that exist in our dataset - NEVER hallucinate or make up properties
            2. For exact matches: ALL user requirements must be precisely met
            3. For similar matches: Location must be EXACTLY the same, other criteria can vary within ±10% for budget, ±1 for bedrooms/bathrooms
            4. If NO properties match the requirements, clearly state this and ask for contact info to notify about future matches
            5. Always be truthful about what's available in our database
            6. When showing property listings, include brief descriptions to help users understand each property's appeal
            7. When users ask for "details", "more info", or mention a specific property ID, provide complete property information

            CONVERSATION APPROACH:
            1. Begin professionally by introducing yourself and ask 1-2 specific questions
            2. Gather key requirements one or two at a time (budget, location, bedrooms, etc.)
            3. Continue asking questions until the user indicates they have no more requirements
            4. Show at most 3 properties at a time with key features, descriptions, and personalized recommendations
            5. When showing properties, mention how to get full details (e.g., "Say 'details [ID]' for complete information")
            6. Provide complete property details when users request more information about specific properties
            7. Request contact info (WhatsApp/email) when: no exact match found, scheduling a viewing, or conversation ending

            PROPERTY DISPLAY FORMAT (for brief listings):
            - Property ID: [ID]
            - Price: $[PRICE]
            - Bedrooms: [NUMBER]
            - Bathrooms: [NUMBER]
            - Square Footage: [SQFT] sq ft
            - Description: [PERSONALIZED DESCRIPTION]
            - Highlight: [ONE STANDOUT FEATURE]
            - Instructions for full details

            FULL DETAILS FORMAT:
            When users request details about a specific property, show:
            - Complete property information including all available fields
            - Location details (street, city, state/zip, country)
            - Construction details (year built, renovated, age)
            - Property features (waterfront, view rating, condition)
            - Space breakdown (above ground, basement, lot size)
            - Special highlights and personalized overview
            - Call to action for viewing

            CONTACT INFO COLLECTION:
            - Always collect WhatsApp/email when no exact matches are found
            - Always collect WhatsApp/email when scheduling property viewings
            - Always collect WhatsApp/email when the conversation appears to be ending
            - Save all user requirements to CSV when contact info is provided
            - Confirm saving and explicitly show saved requirements to the user

            CONVERSATION STAGE: {conversation_stage}
            USER REQUIREMENTS: {user_requirements}
            SEARCH RESULTS: {search_results}
            SELECTED PROPERTY: {selected_property_details}

            Remember: Be honest about what's available, provide helpful descriptions, and never suggest non-existent properties.
            """
            
            # Create prompt templates
            contextual_prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{query}")
            ])
            
            # Define the function to extract user requirements using LLM
            def extract_requirements_with_llm(state: AgentState) -> AgentState:
                if not state.get("user_requirements"):
                    state["user_requirements"] = {}
                
                # Check if user is asking for property details
                query_lower = state["query"].lower()
                
                # Extract property ID if user is asking for details
                property_id_patterns = [
                    r'details?\s+(\d+)',
                    r'more\s+info\s+(\d+)',
                    r'tell\s+me\s+more\s+about\s+(\d+)',
                    r'property\s+(\d+)',
                    r'full\s+details\s+(\d+)',
                    r'information\s+about\s+(\d+)'
                ]
                
                for pattern in property_id_patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        try:
                            state["selected_property_id"] = int(match.group(1))
                            state["conversation_stage"] = "providing_details"
                            return state
                        except ValueError:
                            pass
                
                # Use LLM to extract requirements from the conversation
                extraction_prompt = """
                You are an expert at extracting real estate requirements from user messages.
                Analyze the user's message and previous requirements to extract specific property criteria.
                
                USER MESSAGE: {query}
                CURRENT REQUIREMENTS: {current_requirements}
                
                Extract and update the following information (only include if explicitly mentioned):
                - budget: numerical value (convert k/thousand/million to actual numbers)
                - bedrooms: integer number
                - bathrooms: integer or float number  
                - sqft_living: integer square footage
                - location: city, neighborhood, or area name
                - property_type: apartment, condo, house, townhouse, villa, etc.
                - special_requirements: any other specific needs mentioned
                
                Return ONLY a valid JSON object with the extracted requirements. 
                If a requirement is not mentioned, don't include it in the JSON.
                If updating existing requirements, merge with current ones.
                
                Example: {{"budget": 500000, "bedrooms": 3, "location": "Seattle"}}
                """
                
                try:
                    extraction_response = self.llm.invoke(
                        extraction_prompt.format(
                            query=state["query"],
                            current_requirements=json.dumps(state["user_requirements"])
                        )
                    )
                    
                    # Parse the LLM response to extract JSON
                    response_text = extraction_response.content.strip()
                    
                    # Try to extract JSON from the response
                    try:
                        # Look for JSON object in the response
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            extracted_requirements = json.loads(json_match.group())
                            # Merge with existing requirements
                            state["user_requirements"].update(extracted_requirements)
                        
                    except json.JSONDecodeError:
                        print("Could not parse LLM requirements extraction response as JSON")
                
                except Exception as e:
                    print(f"Error in LLM requirements extraction: {e}")
                
                print("Updated requirements:", state["user_requirements"])
                return state
            
            # Define the function to search for properties in dataset ONLY
            def search_properties_in_dataset(state: AgentState) -> AgentState:
                """Search for properties in the dataset only - no hallucination"""
                # If user is asking for specific property details, skip search
                if state.get("selected_property_id"):
                    return state
                    
                if not state.get("user_requirements"):
                    state["exact_matches"] = []
                    state["similar_matches"] = []
                    state["no_properties_found"] = False
                    return state
                
                requirements = state["user_requirements"]
                exact_matches = []
                similar_matches = []
                
                # Get filter criteria
                budget = requirements.get("budget")
                bedrooms = requirements.get("bedrooms")
                bathrooms = requirements.get("bathrooms")
                sqft = requirements.get("sqft_living")
                location = requirements.get("location")
                property_type = requirements.get("property_type")
                
                print(f"Searching dataset for: Budget=${budget}, Beds={bedrooms}, Baths={bathrooms}, Sqft={sqft}, Location={location}")
                
                # Start with full dataset
                df_copy = self.df.copy()
                
                # STEP 1: Find exact matches with strict criteria
                exact_df = df_copy.copy()
                
                if budget:
                    # Exact budget match (within $1000 tolerance)
                    exact_df = exact_df[abs(exact_df['price'] - budget) <= 1000]
                
                if bedrooms is not None:
                    exact_df = exact_df[exact_df['bedrooms'] == bedrooms]
                
                if bathrooms is not None:
                    exact_df = exact_df[exact_df['bathrooms'] == bathrooms]
                
                if sqft:
                    # Exact sqft match (within 50 sqft tolerance)
                    exact_df = exact_df[abs(exact_df['sqft_living'] - sqft) <= 50]
                
                if location:
                    # Check multiple location columns if they exist
                    location_filter = False
                    if 'city' in exact_df.columns:
                        location_filter |= exact_df['city'].str.lower().str.contains(location.lower(), na=False)
                    if 'neighborhood' in exact_df.columns:
                        location_filter |= exact_df['neighborhood'].str.lower().str.contains(location.lower(), na=False)
                    if 'statezip' in exact_df.columns:
                        location_filter |= exact_df['statezip'].str.lower().str.contains(location.lower(), na=False)
                    
                    if isinstance(location_filter, bool) and not location_filter:
                        # No location columns found, skip location filter for now
                        pass
                    else:
                        exact_df = exact_df[location_filter]
                
                if property_type:
                    if 'property_type' in exact_df.columns:
                        exact_df = exact_df[exact_df['property_type'].str.lower() == property_type.lower()]
                
                # Get exact matches
                if not exact_df.empty:
                    exact_matches = exact_df.head(5).to_dict('records')
                    print(f"Found {len(exact_matches)} exact matches")
                
                # STEP 2: If no exact matches, find similar matches with location constraint
                if len(exact_matches) == 0:
                    similar_df = df_copy.copy()
                    
                    # CRITICAL: For similar matches, location must be EXACTLY the same
                    if location:
                        location_matched = False
                        if 'city' in similar_df.columns:
                            city_match = similar_df['city'].str.lower().str.contains(location.lower(), na=False)
                            if city_match.any():
                                similar_df = similar_df[city_match]
                                location_matched = True
                        
                        if not location_matched and 'neighborhood' in similar_df.columns:
                            neighborhood_match = similar_df['neighborhood'].str.lower().str.contains(location.lower(), na=False)
                            if neighborhood_match.any():
                                similar_df = similar_df[neighborhood_match]
                                location_matched = True
                        
                        if not location_matched and 'statezip' in similar_df.columns:
                            statezip_match = similar_df['statezip'].str.lower().str.contains(location.lower(), na=False)
                            if statezip_match.any():
                                similar_df = similar_df[statezip_match]
                                location_matched = True
                        
                        # If location specified but no properties found in that location
                        if not location_matched:
                            print(f"No properties found in specified location: {location}")
                            similar_df = similar_df.iloc[0:0]  # Empty dataframe
                    
                    # Apply relaxed criteria for other requirements
                    if not similar_df.empty:
                        if budget:
                            # ±10% budget tolerance for similar matches
                            budget_min = budget * 0.9
                            budget_max = budget * 1.1
                            similar_df = similar_df[similar_df['price'].between(budget_min, budget_max)]
                        
                        if bedrooms is not None:
                            # ±1 bedroom tolerance
                            similar_df = similar_df[(similar_df['bedrooms'] >= bedrooms - 1) & 
                                                    (similar_df['bedrooms'] <= bedrooms + 1)]
                        
                        if bathrooms is not None:
                            # ±0.5 bathroom tolerance
                            similar_df = similar_df[(similar_df['bathrooms'] >= bathrooms - 0.5) & 
                                                    (similar_df['bathrooms'] <= bathrooms + 0.5)]
                        
                        if sqft:
                            # ±10% sqft tolerance for similar matches
                            sqft_min = sqft * 0.9
                            sqft_max = sqft * 1.1
                            similar_df = similar_df[similar_df['sqft_living'].between(sqft_min, sqft_max)]
                        
                        if property_type and 'property_type' in similar_df.columns:
                            similar_df = similar_df[similar_df['property_type'].str.lower() == property_type.lower()]
                        
                        # Sort similar matches by price proximity to budget
                        if budget and not similar_df.empty:
                            similar_df['price_diff'] = abs(similar_df['price'] - budget)
                            similar_df = similar_df.sort_values('price_diff')
                        
                        # Get similar matches
                        similar_matches = similar_df.head(5).to_dict('records')
                        print(f"Found {len(similar_matches)} similar matches")
                
                # Update state
                state["exact_matches"] = exact_matches
                state["similar_matches"] = similar_matches
                state["no_properties_found"] = (len(exact_matches) == 0 and len(similar_matches) == 0)
                state["needs_contact_info"] = state["no_properties_found"] or len(exact_matches) == 0
                
                print(f"Search complete - Exact: {len(exact_matches)}, Similar: {len(similar_matches)}, None found: {state['no_properties_found']}")
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
                
                # Check if user is asking for property details
                if state.get("selected_property_id"):
                    state["conversation_stage"] = "providing_details"
                    return state
                
                # Analyze user query for intent
                query = state["query"].lower()
                wants_to_see_properties = any(phrase in query for phrase in [
                    "show me", "property", "properties", "houses", "listing", 
                    "see what", "available", "options", "show property", "find me"
                ])
                
                conversation_ending = any(phrase in query for phrase in [
                    "bye", "goodbye", "thank you", "thanks", "that's all", 
                    "that is all", "end", "finish", "done"
                ])
                
                wants_details = any(phrase in query for phrase in [
                    "tell me more", "more details", "more information", "details", 
                    "features", "about property", "about house", "full details"
                ])
                
                wants_viewing = any(phrase in query for phrase in [
                    "visit", "see in person", "tour", "viewing", "see it", 
                    "appointment", "schedule", "when can i", "available to see"
                ])
                
                # Stage transitions
                if current_stage == "initial_greeting":
                    state["conversation_stage"] = "gathering_requirements"
                
                elif current_stage == "gathering_requirements":
                    # Check if we have enough requirements to search
                    has_basic_requirements = any(key in requirements for key in ["budget", "bedrooms", "location"])
                    
                    if wants_to_see_properties and has_basic_requirements:
                        state["conversation_stage"] = "searching_properties"
                    elif conversation_ending:
                        state["conversation_stage"] = "requesting_contact"
                
                elif current_stage == "searching_properties":
                    if state.get("no_properties_found"):
                        state["conversation_stage"] = "no_matches_found"
                    elif state.get("exact_matches"):
                        state["conversation_stage"] = "showing_exact_matches"
                    elif state.get("similar_matches"):
                        state["conversation_stage"] = "showing_similar_matches"
                    else:
                        state["conversation_stage"] = "no_matches_found"
                
                elif current_stage in ["showing_exact_matches", "showing_similar_matches"]:
                    if wants_details:
                        state["conversation_stage"] = "providing_details"
                    elif wants_viewing:
                        state["conversation_stage"] = "requesting_contact"
                    elif conversation_ending and not has_contact_info:
                        state["conversation_stage"] = "requesting_contact"
                    else:
                        state["conversation_stage"] = "follow_up"
                
                elif current_stage == "providing_details":
                    if wants_viewing:
                        state["conversation_stage"] = "requesting_contact"
                    elif conversation_ending and not has_contact_info:
                        state["conversation_stage"] = "requesting_contact"
                    else:
                        state["conversation_stage"] = "follow_up"
                
                elif current_stage == "no_matches_found":
                    state["conversation_stage"] = "requesting_contact"
                
                elif current_stage == "requesting_contact" and has_contact_info:
                    state["conversation_stage"] = "contact_collected"
                
                # Force contact collection in specific situations
                if (state.get("no_properties_found") or wants_viewing or 
                    (conversation_ending and len(state.get("exact_matches", [])) == 0)):
                    state["needs_contact_info"] = True
                
                return state
            
            # Define the function to collect contact information
            def collect_contact_info(state: AgentState) -> AgentState:
                query = state["query"]
                
                # Initialize contact info if not present
                if "contact_info" not in state:
                    state["contact_info"] = {}
                
                # Extract email using regex
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                email_match = re.search(email_pattern, query)
                if email_match:
                    state["contact_info"]["email"] = email_match.group(0)
                
                # Extract WhatsApp/phone number using regex
                phone_pattern = r'(?:\+?\d{1,4}[\s\-\.])?(?:\(?\d{2,5}\)?)[\s\-\.]?\d{2,5}[\s\-\.]?\d{2,5}(?:[\s\-\.]?\d{1,5})?'
                phone_match = re.search(phone_pattern, query)
                if phone_match:
                    state["contact_info"]["whatsapp"] = phone_match.group(0)
                
                return state
            
            # Define the function to generate response
            def generate_response(state: AgentState) -> AgentState:
                # Format user requirements
                user_requirements_str = "No requirements collected yet."
                if state.get("user_requirements"):
                    requirements = state["user_requirements"]
                    user_requirements_str = "Requirements collected:\n"
                    
                    if "budget" in requirements:
                        user_requirements_str += f"- Budget: ${requirements['budget']:,.0f}\n"
                    if "bedrooms" in requirements:
                        user_requirements_str += f"- Bedrooms: {requirements['bedrooms']}\n"
                    if "bathrooms" in requirements:
                        user_requirements_str += f"- Bathrooms: {requirements['bathrooms']}\n"
                    if "sqft_living" in requirements:
                        user_requirements_str += f"- Square Footage: {requirements['sqft_living']} sq ft\n"
                    if "location" in requirements:
                        user_requirements_str += f"- Location: {requirements['location']}\n"
                    if "property_type" in requirements:
                        user_requirements_str += f"- Property Type: {requirements['property_type']}\n"
                
                # Format search results
                search_results_str = ""
                
                # Handle specific property details request
                selected_property_details = ""
                if state.get("selected_property_id") and self.property_index:
                    property_id = state["selected_property_id"]
                    if property_id in self.property_index:
                        prop = self.property_index[property_id]
                        selected_property_details = self.format_property_full_details(prop)
                        search_results_str = f"USER REQUESTED FULL DETAILS FOR PROPERTY ID {property_id}:\n{selected_property_details}"
                    else:
                        search_results_str = f"PROPERTY ID {property_id} NOT FOUND IN DATABASE.\n"
                
                # Add exact matches if found (for brief listings)
                elif state.get("exact_matches"):
                    search_results_str += "EXACT MATCHES FOUND IN DATABASE:\n"
                    for i, prop in enumerate(state["exact_matches"][:3]):  # Show max 3
                        search_results_str += self.format_property_brief(prop, i+1)
                
                # Add similar matches if no exact matches (for brief listings)
                elif state.get("similar_matches"):
                    search_results_str += "SIMILAR MATCHES FOUND IN DATABASE (within criteria):\n"
                    for i, prop in enumerate(state["similar_matches"][:3]):  # Show max 3
                        search_results_str += self.format_property_brief(prop, i+1)
                
                # Handle no matches found
                elif state.get("no_properties_found"):
                    search_results_str += "NO PROPERTIES FOUND IN DATABASE matching the specified requirements.\n"
                    search_results_str += "We need to collect contact information to notify about future matches.\n"
                
                # Add contact info status
                contact_info = state.get("contact_info", {})
                if contact_info.get("email") or contact_info.get("whatsapp"):
                    search_results_str += f"\nCONTACT INFO PROVIDED: Email: {contact_info.get('email', 'Not provided')}, WhatsApp: {contact_info.get('whatsapp', 'Not provided')}\n"
                elif state.get("needs_contact_info"):
                    search_results_str += "\nACTION NEEDED: Request contact information (email or WhatsApp) to save requirements.\n"
                
                # Generate response using LLM
                response = self.llm.invoke(
                    contextual_prompt.invoke({
                        "search_results": search_results_str,
                        "user_requirements": user_requirements_str,
                        "conversation_stage": state["conversation_stage"],
                        "selected_property_details": selected_property_details,
                        "chat_history": state.get("chat_history", []),
                        "query": state["query"]
                    })
                )
                
                state["response"] = response.content
                return state
            
            # Create the LangGraph workflow
            workflow = StateGraph(AgentState)
            
            # Define the nodes
            workflow.add_node("extract_requirements", extract_requirements_with_llm)
            workflow.add_node("search_properties", search_properties_in_dataset)
            workflow.add_node("decide_next_action", decide_next_action)
            workflow.add_node("collect_contact_info", collect_contact_info)
            workflow.add_node("generate_response", generate_response)
            
            # Define the edges
            workflow.add_edge("extract_requirements", "search_properties")
            workflow.add_edge("search_properties", "decide_next_action")
            workflow.add_edge("decide_next_action", "collect_contact_info")
            workflow.add_edge("collect_contact_info", "generate_response")
            
            # Set the entry point
            workflow.set_entry_point("extract_requirements")
            
            # Compile the graph
            self.chain = workflow.compile()
            
            print("Enhanced chain setup complete!")
            return self.chain
            
        except Exception as e:
            print(f"Error setting up chain: {e}")
            return None

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
                "exact_matches": result.get("exact_matches", []),
                "similar_matches": result.get("similar_matches", []),
                "needs_contact_info": result.get("needs_contact_info", False),
                "no_properties_found": result.get("no_properties_found", False),
                "requirements_saved": result.get("requirements_saved", False),
                "selected_property_id": result.get("selected_property_id", None)
            }
        }

        # Save requirements only when contact info is provided and not already saved
        if (response["system_state"]["contact_info"].get("email") or 
            response["system_state"]["contact_info"].get("whatsapp")) and not response["system_state"]["requirements_saved"]:
            
            # Prepare data for saving
            save_data = {
                "user_requirements": response["system_state"]["requirements"],
                "contact_info": response["system_state"]["contact_info"],
                "conversation_stage": response["system_state"]["conversation_stage"]
            }
            
            # Convert numpy types before saving
            save_data = convert_numpy_types(save_data)
            
            save_success = save_user_requirements(save_data)
            response["system_state"]["requirements_saved"] = save_success
            
            if save_success:
                print("User requirements saved successfully to CSV")

        # Add property data to response for reference (but don't expose in conversation unless from dataset)
        if "exact_matches" in result and result["exact_matches"]:
            response["exact_matches"] = result["exact_matches"][:3]  # Limit to 3

        if "similar_matches" in result and result["similar_matches"]:
            response["similar_matches"] = result["similar_matches"][:3]  # Limit to 3

        return response

    except Exception as e:
        print(f"Error processing query: {e}")
        return {
            "answer": "I apologize, but I encountered an error while searching our property database. Please try rephrasing your request or contact our support team.",
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
            }
        }