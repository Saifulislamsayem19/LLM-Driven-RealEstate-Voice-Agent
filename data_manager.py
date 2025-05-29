import os
import pandas as pd
import numpy as np
import hashlib
import time
import uuid
from datetime import datetime
from typing import Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

VECTOR_DB_DIR = "vector_db"
USER_DATA_DIR = "user_data"
dataset_path = "dataset/USA Housing Dataset.csv"
USER_REQUIREMENTS_FILE = os.path.join(USER_DATA_DIR, "user_requirements.csv")
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

def get_db_hash(file_path: str, mtime: float) -> str:
    """Generate a hash for the database based on file path and modification time"""
    hash_input = f"{file_path}_{mtime}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def create_documents(df_text, df_original):
    """Create documents for vectorization from the dataframe"""
    documents = []
    
    # Create overall dataset summary
    dataset_summary = f"""
    Real Estate Dataset Summary:
    Total Properties: {len(df_original)}
    Price Range: ${df_original['price'].min()} to ${df_original['price'].max()}
    Average Price: ${df_original['price'].mean():.2f}
    Average Square Footage: {df_original['sqft_living'].mean():.2f} sq ft
    Features tracked: {', '.join(df_original.columns)}
    """
    
    summary_doc = Document(
        page_content=dataset_summary,
        metadata={
            "document_type": "summary",
            "source": "dataset_summary"
        }
    )
    documents.append(summary_doc)
    
    # Create individual property documents
    for idx, row in df_text.iterrows():
        property_id = idx
        
        # Format property data in a structured way
        property_text = f"""
        Property ID: {property_id}
        Price: ${row['price'] if 'price' in row else 'N/A'}
        Bedrooms: {row['bedrooms'] if 'bedrooms' in row else 'N/A'}
        Bathrooms: {row['bathrooms'] if 'bathrooms' in row else 'N/A'}
        Square Footage (Living): {row['sqft_living'] if 'sqft_living' in row else 'N/A'} sq ft
        Square Footage (Lot): {row['sqft_lot'] if 'sqft_lot' in row else 'N/A'} sq ft
        Floors: {row['floors'] if 'floors' in row else 'N/A'}
        Waterfront: {"Yes" if row.get('waterfront', 0) == 1 else "No"}
        View Quality: {row['view'] if 'view' in row else 'N/A'} out of 5
        Condition: {row['condition'] if 'condition' in row else 'N/A'} out of 5
        Year Built: {row['yr_built'] if 'yr_built' in row else 'N/A'}
        Year Renovated: {row['yr_renovated'] if 'yr_renovated' in row and row['yr_renovated'] > 0 else "Not renovated"}
        """
        
        # Add additional fields if they exist
        if 'zipcode' in row:
            property_text += f"Zipcode: {row['zipcode']}\n"
        if 'city' in row:
            property_text += f"City: {row['city']}\n"
        if 'street' in row:
            property_text += f"Street: {row['street']}\n"
        if 'neighborhood' in row:
            property_text += f"Neighborhood: {row['neighborhood']}\n"
            
        # Create metadata for filtering
        metadata = {
            "document_type": "property",
            "property_id": property_id,
            "price": float(df_original.iloc[idx]['price']) if 'price' in df_original.columns else None,
            "bedrooms": float(df_original.iloc[idx]['bedrooms']) if 'bedrooms' in df_original.columns else None,
            "bathrooms": float(df_original.iloc[idx]['bathrooms']) if 'bathrooms' in df_original.columns else None,
            "sqft_living": float(df_original.iloc[idx]['sqft_living']) if 'sqft_living' in df_original.columns else None
        }
        
        doc = Document(page_content=property_text, metadata=metadata)
        documents.append(doc)
    
    return documents

def load_and_preprocess_data():
    """Load and preprocess the housing dataset"""
    try:
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file not found at {dataset_path}")
            return None, None, None
        
        # Get file modification time for cache invalidation
        file_mtime = os.path.getmtime(dataset_path)
        
        # Load the CSV file
        df = pd.read_csv(dataset_path)
        
        # Clean column names (remove spaces, etc.)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # Handle missing values
        df = df.replace(['NA', 'N/A', 'None', 'NULL', ''], np.nan)
        
        # Remove rows where 'price' is NaN
        df = df.dropna(subset=['price'])
        
        # Remove rows where 'price' is 0
        df = df[df['price'] != 0]

        # Reset index to ensure contiguous indices
        df = df.reset_index(drop=True)
        
        # Convert numeric columns to appropriate types
        numeric_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                        'floors', 'waterfront', 'view', 'condition', 'sqft_above',
                        'sqft_basement', 'yr_built', 'yr_renovated']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Fill missing values with appropriate placeholder text for better context
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("Information not available")
            else:
                df[col] = df[col].fillna(-1)  # For numeric columns
            
        # Convert -1 values back to readable format in the text representation
        df_for_text = df.copy()
        for col in df_for_text.columns:
            if df_for_text[col].dtype != 'object':
                df_for_text[col] = df_for_text[col].apply(lambda x: "Information not available" if x == -1 else x)
        
        # Add a unique property ID column if not present
        if 'property_id' not in df.columns:
            df['property_id'] = df.index
        
        # Build property index for quick lookups
        property_index = {row['property_id']: row.to_dict() for _, row in df.iterrows()}
        df = df
        
        return df, df_for_text, file_mtime
    
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        return None, None, None

def setup_vector_db(df, df_for_text, file_mtime):
    """Create or load vector database with improved chunking"""
    
    if df is None:
        return None, None
    
    # Generate hash for this dataset version
    db_hash = get_db_hash(dataset_path, file_mtime)
    db_path = os.path.join(VECTOR_DB_DIR, f"faiss_index_{db_hash}")
    
    # Check if we have this version cached
    if os.path.exists(db_path):
        print(f"Loading existing vector database from {db_path}")
        try:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            print("Vector database loaded successfully!")
            
            vector_store = vector_store
            return df, vector_store
        except Exception as e:
            print(f"Error loading vector database: {e}")
            print("Creating new vector database...")
    else:
        print("No matching vector database found. Creating new one...")
    
    # Create documents
    print("Creating document chunks for the vector database...")
    documents = create_documents(df_for_text, df)
    
    # Define better text splitting strategy based on document type
    property_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    summary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    # Process documents based on their type
    texts = []
    for doc in documents:
        if doc.metadata.get("document_type") == "summary":
            chunk_docs = summary_splitter.split_documents([doc])
            texts.extend(chunk_docs)
        else:
            chunk_docs = property_splitter.split_documents([doc])
            texts.extend(chunk_docs)
    
    # Create embeddings and vector store
    print("Creating vector database with OpenAI embeddings...")
    total_chunks = len(texts)
    print(f"Total chunks to process: {total_chunks}")
    
    embeddings = OpenAIEmbeddings()
    
    # Process in batches to show progress
    start_time = time.time()
    
    # Create vector store with batched processing
    batch_size = 100
    all_texts = texts
    
    # Initialize progress tracking
    processed = 0
    total = len(all_texts)
    
    # Process in batches and show progress
    batches = [all_texts[i:i + batch_size] for i in range(0, len(all_texts), batch_size)]
    
    for i, batch in enumerate(batches):
        batch_start = time.time()
        
        # For the first batch, create the vector store
        if i == 0:
            vector_store = FAISS.from_documents(batch, embeddings)
        # For subsequent batches, add to existing vector store
        else:
            vector_store.add_documents(batch)
            
        processed += len(batch)
        batch_time = time.time() - batch_start
        
        # Show progress
        progress = (processed / total) * 100
        print(f"Progress: {processed}/{total} chunks ({progress:.2f}%) - Batch {i+1}/{len(batches)} processed in {batch_time:.2f}s")
    
    elapsed_time = time.time() - start_time
    print(f"Vector database creation completed in {elapsed_time:.2f} seconds!")
    
    # Save the vector store
    print(f"Saving vector database to {db_path}")
    vector_store.save_local(db_path)
    
    vector_store = vector_store
    return df, vector_store

def convert_numpy_types(obj):
    """Recursively convert numpy and pandas types to Python native types"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # Convert Timestamp to ISO format string
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_user_requirements(state: Dict[str, Any]):
    """Save user requirements and contact information to CSV file in specific sequence"""
    try:
        # Don't save if no contact info or no requirements
        contact_info = convert_numpy_types(state.get("contact_info", {}))
        requirements = convert_numpy_types(state.get("user_requirements", {}))
        
        if not contact_info.get("email") and not contact_info.get("whatsapp"):
            print("No contact information provided, skipping save")
            return False
            
        if not requirements:
            print("No requirements collected, skipping save")
            return False
            
        # Create a dictionary with all the information to save in the specific required sequence
        user_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": str(uuid.uuid4())[:8],  # Generate a simple user ID
            "email": contact_info.get("email", ""),
            "whatsapp": contact_info.get("whatsapp", ""),
            "price_min": requirements.get("budget", ""),
            "bedrooms": requirements.get("bedrooms", ""),
            "bathrooms": requirements.get("bathrooms", ""),
            "sqft_living": requirements.get("sqft_living", ""),
            "location": requirements.get("location", ""),
            "property_type": requirements.get("property_type", ""),
            "specific_requirements": requirements.get("special_requirements", "")
        }

        # Convert to DataFrame for CSV storage
        df_row = pd.DataFrame([convert_numpy_types(user_data)])

        # Check if file exists to determine if we need headers
        file_exists = os.path.isfile(USER_REQUIREMENTS_FILE)

        # Append to CSV
        if file_exists:
            # If file exists but doesn't have the correct columns, rewrite with headers
            try:
                existing_df = pd.read_csv(USER_REQUIREMENTS_FILE)
                if list(existing_df.columns) != list(df_row.columns):
                    # Columns don't match expected sequence, create new file with correct headers
                    df_row.to_csv(USER_REQUIREMENTS_FILE, mode='w', header=True, index=False)
                else:
                    # Columns match, append without headers
                    df_row.to_csv(USER_REQUIREMENTS_FILE, mode='a', header=False, index=False)
            except Exception:
                # If reading fails, create new file
                df_row.to_csv(USER_REQUIREMENTS_FILE, mode='w', header=True, index=False)
        else:
            # File doesn't exist, create with headers
            df_row.to_csv(USER_REQUIREMENTS_FILE, mode='w', header=True, index=False)

        print(f"User requirements saved to {USER_REQUIREMENTS_FILE}")
        # Set flag in state to confirm saving
        state["requirements_saved"] = True
        return True
        
    except Exception as e:
        print(f"Error saving user requirements: {e}")
        return False
