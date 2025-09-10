import re
import json
from typing import Dict, List, Any, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from data_manager import load_and_preprocess_data, setup_vector_db, save_user_requirements, convert_numpy_types

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
        # Unpack both returned values
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
                temperature=0.3
            )
            
            # System template for conversation
            system_template = """You are a professional AI real estate agent assistant named HomeFind AI. 
            Your job is to help users find their ideal home based on their specific requirements.

            CRITICAL RULES:
            1. ONLY suggest properties that exist in our dataset - NEVER hallucinate or make up properties
            2. MANDATORY: Users must provide both BUDGET AND LOCATION before showing any properties
            3. For exact matches: ALL user requirements must be precisely met
            4. For similar matches: Location must be EXACTLY the same, other criteria can vary within specified tolerances
            5. If NO properties match the requirements, clearly state this and ask for contact info
            6. Always be truthful about what's available in our database
            7. When users ask for "details", "more info", or mention property by number or ID, provide complete property information

            BUDGET AND LOCATION REQUIREMENT:
            - If user asks to see properties without providing budget, ask for their budget range
            - If user asks to see properties without providing location, ask for their preferred location/area
            - If user provides only one of budget OR location, ask for the missing one
            - Only search and show properties after BOTH budget and location are provided

            PROPERTY DETAILS HANDLING:
            - Handle requests like "details for property 1", "more info about property 2", "tell me about property ID 12345"
            - Extract both property numbers (from listings) and property IDs
            - Convert property numbers to actual property IDs using the shown properties list

            CONVERSATION APPROACH:
            1. Begin professionally and ask for budget and location if not provided
            2. Gather additional requirements (bedrooms, bathrooms, sqft, special features)
            3. Only show properties when budget AND location are confirmed
            4. Show at most 3 properties at a time with key features and descriptions
            5. Provide complete details when requested for specific properties
            6. Request contact info when: no matches found, scheduling viewing, or conversation ending

            COMPREHENSIVE FILTERING CRITERIA:
            - Budget: price range filtering
            - Location: city, neighborhood, state/zip matching
            - Bedrooms: exact number or range
            - Bathrooms: exact number or range
            - Square footage: living space size filtering
            - Property type: house, condo, apartment, etc.
            - Year built: construction year filtering
            - Year renovated: renovation year filtering
            - Waterfront: waterfront properties (yes/no)
            - View rating: scenic view quality (1-5 scale)
            - Condition rating: property condition (1-5 scale)
            - Floors: number of floors/levels
            - Lot size: outdoor space size
            - Basement: basement square footage
            - Special features: garage, fireplace, etc.

            CONVERSATION STAGE: {conversation_stage}
            USER REQUIREMENTS: {user_requirements}
            SEARCH RESULTS: {search_results}
            SELECTED PROPERTY: {selected_property_details}
            PROPERTIES SHOWN: {properties_shown}

            Remember: No properties shown without budget AND location. Be honest about availability.
            """
            
            # Create prompt templates
            contextual_prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{query}")
            ])
            
            # Enhanced requirements extraction with comprehensive filtering
            def extract_requirements_with_llm(state: AgentState) -> AgentState:
                if not state.get("user_requirements"):
                    state["user_requirements"] = {}
                
                # Check if user is asking for property details
                query_lower = state["query"].lower()
                
                # Enhanced property detail request patterns
                property_detail_patterns = [
                    # Direct property ID patterns
                    r'(?:details?|info|information|more)\s+(?:about|for|on)?\s*(?:property\s+)?(?:id\s+)?(\d+)',
                    r'(?:tell\s+me\s+(?:more\s+)?about|show\s+me\s+(?:more\s+)?(?:details?|info))\s+(?:property\s+)?(?:id\s+)?(\d+)',
                    r'(?:property\s+)?(?:id\s+)?(\d+)\s+(?:details?|info|information)',
                    r'(?:full\s+)?(?:details?|info|information)\s+(?:of|for|about)\s+(?:property\s+)?(?:id\s+)?(\d+)',
                    r'(?:details?|info|information|more)\s+(?:about|for|on)?\s*(?:property\s+)?(?:number\s+)?(\d+)(?:st|nd|rd|th)?\s*$',
                    r'(?:tell\s+me\s+(?:more\s+)?about|show\s+me\s+(?:more\s+)?(?:details?|info))\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+(?:property|one)',
                    r'(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+(?:property|one)',
                    r'property\s+(?:number\s+)?(\d+)(?:st|nd|rd|th)?',
                    r'(\d+)(?:st|nd|rd|th)?\s+property'
                ]
                
                for pattern in property_detail_patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        try:
                            requested_number = int(match.group(1))
                            
                            # Check if it's a property number (1-3) from recent listings
                            if 1 <= requested_number <= 3 and state.get("properties_shown"):
                                # Convert property number to actual property ID
                                property_index = requested_number - 1
                                if property_index < len(state["properties_shown"]):
                                    state["selected_property_id"] = state["properties_shown"][property_index]
                                    state["conversation_stage"] = "providing_details"
                                    return state
                            
                            # Otherwise treat as direct property ID
                            state["selected_property_id"] = requested_number
                            state["conversation_stage"] = "providing_details"
                            return state
                        except ValueError:
                            pass
                
                # Enhanced LLM extraction for comprehensive filtering
                extraction_prompt = """
                You are an expert at extracting comprehensive real estate requirements from user messages.
                Analyze the user's message and previous requirements to extract ALL possible property criteria.
                
                USER MESSAGE: {query}
                CURRENT REQUIREMENTS: {current_requirements}
                
                Extract and update ALL mentioned requirements (only include if explicitly mentioned):
                
                BASIC REQUIREMENTS:
                - budget: numerical value (convert k/thousand/million to actual numbers)
                - location: city, neighborhood, area, state name
                - bedrooms: integer number
                - bathrooms: integer or float number
                - sqft_living: integer square footage of living space
                
                ADVANCED FILTERING:
                - property_type: house, condo, apartment, townhouse, villa, etc.
                - year_built: construction year (YYYY format)
                - year_renovated: renovation year (YYYY format)
                - waterfront: true/false for waterfront properties
                - view_rating: integer 1-5 for scenic views
                - condition_rating: integer 1-5 for property condition
                - floors: integer number of floors/levels
                - sqft_lot: integer lot/yard size in square feet
                - sqft_basement: integer basement size in square feet
                - garage: true/false or number of garage spaces
                - fireplace: true/false
                - swimming_pool: true/false
                - garden: true/false
                - parking: true/false or number of spaces
                - furnished: true/false
                - pets_allowed: true/false
                - new_construction: true/false (built within last 5 years)
                - recently_renovated: true/false (renovated within last 10 years)
                
                PRICE RANGES:
                - budget_min: minimum price if range given
                - budget_max: maximum price if range given
                
                SIZE RANGES:
                - bedrooms_min: minimum bedrooms if range given
                - bedrooms_max: maximum bedrooms if range given
                - bathrooms_min: minimum bathrooms if range given
                - bathrooms_max: maximum bathrooms if range given
                - sqft_living_min: minimum living space if range given
                - sqft_living_max: maximum living space if range given
                
                SPECIAL REQUIREMENTS:
                - special_features: list of any other specific requirements mentioned
                
                Return ONLY a valid JSON object with the extracted requirements.
                If a requirement is not mentioned, don't include it in the JSON.
                If a requirement is in string or ambiguous format, convert to appropriate type.
                If a requirement contradicts previous ones, update to the latest.
                Merge with existing requirements without overriding unless explicitly contradicted.
                
                Example: {{"budget": 500000, "location": "Seattle", "bedrooms": 3, "waterfront": true, "garage": true}}
                """
                
                try:
                    extraction_response = self.llm.invoke(
                        extraction_prompt.format(
                            query=state["query"],
                            current_requirements=json.dumps(state["user_requirements"])
                        )
                    )
                    
                    response_text = extraction_response.content.strip()
                    
                    try:
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            extracted_requirements = json.loads(json_match.group())
                            state["user_requirements"].update(extracted_requirements)
                        
                    except json.JSONDecodeError:
                        print("Could not parse LLM requirements extraction response as JSON")
                
                except Exception as e:
                    print(f"Error in LLM requirements extraction: {e}")
                
                print("Updated requirements:", state["user_requirements"])
                return state
            
            # Enhanced property search with comprehensive filtering
            def search_properties_in_dataset(state: AgentState) -> AgentState:
                """Search for properties with comprehensive filtering - dataset only"""
                # If user is asking for specific property details, skip search
                if state.get("selected_property_id"):
                    return state
                    
                if not state.get("user_requirements"):
                    state["exact_matches"] = []
                    state["similar_matches"] = []
                    state["no_properties_found"] = False
                    return state
                
                requirements = state["user_requirements"]
                
                # CRITICAL: Check for mandatory budget and location
                budget = requirements.get("budget")
                budget_min = requirements.get("budget_min", budget)
                budget_max = requirements.get("budget_max", budget)
                location = requirements.get("location")
                
                # If budget or location missing, don't search
                if not budget and not (budget_min or budget_max):
                    state["exact_matches"] = []
                    state["similar_matches"] = []
                    state["no_properties_found"] = False
                    state["conversation_stage"] = "gathering_requirements"
                    return state
                
                if not location:
                    state["exact_matches"] = []
                    state["similar_matches"] = []
                    state["no_properties_found"] = False
                    state["conversation_stage"] = "gathering_requirements"
                    return state
                
                exact_matches = []
                similar_matches = []
                
                print(f"Comprehensive search for: Budget=${budget}, Location={location}")
                print(f"Additional filters: {requirements}")
                
                # Start with full dataset
                df_copy = self.df.copy()
                
                # STEP 1: Find exact matches with comprehensive filtering
                exact_df = df_copy.copy()
                
                # Budget filtering
                if budget:
                    exact_df = exact_df[abs(exact_df['price'] - budget) <= 5000]  
                elif budget_min or budget_max:
                    if budget_min:
                        exact_df = exact_df[exact_df['price'] >= budget_min]
                    if budget_max:
                        exact_df = exact_df[exact_df['price'] <= budget_max]
                
                # Location filtering
                if location:
                    location_filter = False
                    for col in ['city', 'neighborhood', 'statezip', 'state', 'area']:
                        if col in exact_df.columns:
                            location_filter |= exact_df[col].str.lower().str.contains(location.lower(), na=False)
                    if isinstance(location_filter, bool) and not location_filter:
                        pass
                    else:
                        exact_df = exact_df[location_filter]
                
                # Comprehensive filtering
                filters = [
                    ('bedrooms', 'bedrooms', 0),
                    ('bathrooms', 'bathrooms', 0),
                    ('sqft_living', 'sqft_living', 100),
                    ('property_type', 'property_type', None),
                    ('year_built', 'yr_built', 2),
                    ('year_renovated', 'yr_renovated', 2),
                    ('waterfront', 'waterfront', None),
                    ('view_rating', 'view', 0),
                    ('condition_rating', 'condition', 0),
                    ('floors', 'floors', 0),
                    ('sqft_lot', 'sqft_lot', 500),
                    ('sqft_basement', 'sqft_basement', 100)
                ]
                
                for req_key, df_col, tolerance in filters:
                    if req_key in requirements and df_col in exact_df.columns:
                        value = requirements[req_key]
                        if tolerance is None:
                            # Exact match for categorical data
                            if isinstance(value, bool):
                                exact_df = exact_df[exact_df[df_col] == (1 if value else 0)]
                            elif isinstance(value, str):
                                exact_df = exact_df[exact_df[df_col].str.lower() == value.lower()]
                            else:
                                exact_df = exact_df[exact_df[df_col] == value]
                        elif tolerance == 0:
                            # Exact numeric match
                            exact_df = exact_df[exact_df[df_col] == value]
                        else:
                            # Tolerance-based numeric match
                            exact_df = exact_df[abs(exact_df[df_col] - value) <= tolerance]
                
                # Range filtering
                range_filters = [
                    ('bedrooms_min', 'bedrooms_max', 'bedrooms'),
                    ('bathrooms_min', 'bathrooms_max', 'bathrooms'),
                    ('sqft_living_min', 'sqft_living_max', 'sqft_living')
                ]
                
                for min_key, max_key, df_col in range_filters:
                    if df_col in exact_df.columns:
                        if min_key in requirements:
                            exact_df = exact_df[exact_df[df_col] >= requirements[min_key]]
                        if max_key in requirements:
                            exact_df = exact_df[exact_df[df_col] <= requirements[max_key]]
                
                # Get exact matches
                if not exact_df.empty:
                    exact_matches = exact_df.head(5).to_dict('records')
                    # Store property IDs for reference
                    state["properties_shown"] = [prop['property_id'] for prop in exact_matches[:3]]
                    print(f"Found {len(exact_matches)} exact matches")
                
                # STEP 2: Similar matches with relaxed criteria
                if len(exact_matches) == 0:
                    similar_df = df_copy.copy()
                    
                    # Location must be EXACTLY the same for similar matches
                    if location:
                        location_matched = False
                        for col in ['city', 'neighborhood', 'statezip', 'state', 'area']:
                            if col in similar_df.columns:
                                location_match = similar_df[col].str.lower().str.contains(location.lower(), na=False)
                                if location_match.any():
                                    similar_df = similar_df[location_match]
                                    location_matched = True
                                    break
                        
                        if not location_matched:
                            similar_df = similar_df.iloc[0:0]  
                    
                    # Relaxed filtering for similar matches
                    if not similar_df.empty:
                        # Budget with ±15% tolerance
                        if budget:
                            budget_min_sim = budget * 0.85
                            budget_max_sim = budget * 1.15
                            similar_df = similar_df[similar_df['price'].between(budget_min_sim, budget_max_sim)]
                        elif budget_min or budget_max:
                            if budget_min:
                                similar_df = similar_df[similar_df['price'] >= budget_min * 0.9]
                            if budget_max:
                                similar_df = similar_df[similar_df['price'] <= budget_max * 1.1]
                        
                        # Relaxed criteria for other filters
                        relaxed_filters = [
                            ('bedrooms', 'bedrooms', 1),
                            ('bathrooms', 'bathrooms', 1),
                            ('sqft_living', 'sqft_living', 0.15),  
                            ('year_built', 'yr_built', 10),
                            ('floors', 'floors', 1)
                        ]
                        
                        for req_key, df_col, tolerance in relaxed_filters:
                            if req_key in requirements and df_col in similar_df.columns:
                                value = requirements[req_key]
                                if req_key == 'sqft_living':
                                    # Percentage-based tolerance
                                    min_val = value * (1 - tolerance)
                                    max_val = value * (1 + tolerance)
                                    similar_df = similar_df[similar_df[df_col].between(min_val, max_val)]
                                else:
                                    # Fixed tolerance
                                    similar_df = similar_df[abs(similar_df[df_col] - value) <= tolerance]
                        
                        # Sort by price proximity to budget
                        if budget and not similar_df.empty:
                            similar_df['price_diff'] = abs(similar_df['price'] - budget)
                            similar_df = similar_df.sort_values('price_diff')
                        
                        similar_matches = similar_df.head(5).to_dict('records')
                        # Store property IDs for reference
                        state["properties_shown"] = [prop['property_id'] for prop in similar_matches[:3]]
                        print(f"Found {len(similar_matches)} similar matches")
                
                # Update state
                state["exact_matches"] = exact_matches
                state["similar_matches"] = similar_matches
                state["no_properties_found"] = (len(exact_matches) == 0 and len(similar_matches) == 0)
                state["needs_contact_info"] = state["no_properties_found"]
                
                print(f"Search complete - Exact: {len(exact_matches)}, Similar: {len(similar_matches)}")
                return state
            
            # Enhanced decision logic
            def decide_next_action(state: AgentState) -> AgentState:
                if "conversation_stage" not in state:
                    state["conversation_stage"] = "initial_greeting"
                    return state
                
                current_stage = state["conversation_stage"]
                requirements = state.get("user_requirements", {})
                has_contact_info = "contact_info" in state and (
                    state["contact_info"].get("email") or state["contact_info"].get("whatsapp")
                )
                
                # Check for budget and location requirements
                has_budget = requirements.get("budget") or requirements.get("budget_min") or requirements.get("budget_max")
                has_location = requirements.get("location")
                
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
                
                # Stage transitions with budget/location enforcement
                if current_stage == "initial_greeting":
                    state["conversation_stage"] = "gathering_requirements"
                
                elif current_stage == "gathering_requirements":
                    # Must have both budget and location before showing properties
                    if wants_to_see_properties:
                        if has_budget and has_location:
                            state["conversation_stage"] = "searching_properties"
                        else:
                            # Stay in gathering requirements if missing budget or location
                            state["conversation_stage"] = "gathering_requirements"
                    elif conversation_ending:
                        state["conversation_stage"] = "requesting_contact"
                
                elif current_stage == "searching_properties":
                    if not has_budget or not has_location:
                        state["conversation_stage"] = "gathering_requirements"
                    elif state.get("no_properties_found"):
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
            
            # Contact info collection function
            def collect_contact_info(state: AgentState) -> AgentState:
                query = state["query"]
                
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
            
            # Enhanced response generation
            def generate_response(state: AgentState) -> AgentState:
                # Format user requirements
                user_requirements_str = "No requirements collected yet."
                if state.get("user_requirements"):
                    requirements = state["user_requirements"]
                    user_requirements_str = "Requirements collected:\n"
                    
                    # Basic requirements
                    if "budget" in requirements:
                        user_requirements_str += f"- Budget: ${requirements['budget']:,.0f}\n"
                    elif "budget_min" in requirements or "budget_max" in requirements:
                        min_budget = requirements.get('budget_min', 'No min')
                        max_budget = requirements.get('budget_max', 'No max')
                        user_requirements_str += f"- Budget Range: ${min_budget} - ${max_budget}\n"
                    
                    if "location" in requirements:
                        user_requirements_str += f"- Location: {requirements['location']}\n"
                    if "bedrooms" in requirements:
                        user_requirements_str += f"- Bedrooms: {requirements['bedrooms']}\n"
                    if "bathrooms" in requirements:
                        user_requirements_str += f"- Bathrooms: {requirements['bathrooms']}\n"
                    if "sqft_living" in requirements:
                        user_requirements_str += f"- Square Footage: {requirements['sqft_living']} sq ft\n"
                    
                    # Advanced filters
                    advanced_filters = {
                        'property_type': 'Property Type',
                        'year_built': 'Year Built',
                        'year_renovated': 'Year Renovated',
                        'waterfront': 'Waterfront',
                        'view_rating': 'View Rating',
                        'condition_rating': 'Condition Rating',
                        'floors': 'Floors',
                        'sqft_lot': 'Lot Size (sq ft)',
                        'sqft_basement': 'Basement Size (sq ft)',
                        'garage': 'Garage',
                        'fireplace': 'Fireplace',
                        'swimming_pool': 'Swimming Pool'
                    }
                    
                    for key, label in advanced_filters.items():
                        if key in requirements:
                            value = requirements[key]
                            if isinstance(value, bool):
                                value = "Yes" if value else "No"
                            user_requirements_str += f"- {label}: {value}\n"
                
                # Check if budget and location are present
                has_budget = requirements.get("budget") or requirements.get("budget_min") or requirements.get("budget_max") if requirements else False
                has_location = requirements.get("location") if requirements else False
                
                # Format search results with budget/location check
                search_results_str = ""
                properties_shown_str = ""
                
                # Add property IDs shown for reference
                if state.get("properties_shown"):
                    properties_shown_str = f"Properties currently shown (IDs): {state['properties_shown']}\n"
                
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
                
                # Only show properties if budget AND location are provided
                elif has_budget and has_location:
                    # Add exact matches if found
                    if state.get("exact_matches"):
                        search_results_str += "EXACT MATCHES FOUND IN DATABASE:\n"
                        for i, prop in enumerate(state["exact_matches"][:3]):
                            search_results_str += self.format_property_brief(prop, i+1)
                    
                    # Add similar matches if no exact matches
                    elif state.get("similar_matches"):
                        search_results_str += "SIMILAR MATCHES FOUND IN DATABASE (within criteria):\n"
                        for i, prop in enumerate(state["similar_matches"][:3]):
                            search_results_str += self.format_property_brief(prop, i+1)
                    
                    # Handle no matches found
                    elif state.get("no_properties_found"):
                        search_results_str += "NO PROPERTIES FOUND IN DATABASE matching the specified requirements.\n"
                        search_results_str += "We need to collect contact information to notify about future matches.\n"
                
                else:
                    # Missing budget or location
                    missing_items = []
                    if not has_budget:
                        missing_items.append("BUDGET")
                    if not has_location:
                        missing_items.append("LOCATION")
                    
                    search_results_str += f"MISSING REQUIRED INFORMATION: {' and '.join(missing_items)}\n"
                    search_results_str += "Cannot search properties without both budget and location.\n"
                
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
                        "properties_shown": properties_shown_str,
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
    """Process a user query and generate a response with enhanced filtering and validation"""
    
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
            "answer": "I apologize, but I encountered an error while searching our property database with your comprehensive criteria. Please try rephrasing your request or contact our support team.",
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