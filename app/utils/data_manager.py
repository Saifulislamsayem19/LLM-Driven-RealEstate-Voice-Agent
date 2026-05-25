"""
Data loading, preprocessing, vector DB management, and user-requirement persistence.

All logic preserved exactly from the original data_manager.py.
Only configuration references updated to use settings.
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


# ── Type conversion helper ────────────────────────────────────────────────────

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to Python-native types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_db_hash(mtime: float) -> str:
    hash_input = f"{settings.DATASET_PATH}_{mtime}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def _create_documents(df_text: pd.DataFrame, df_original: pd.DataFrame) -> list[Document]:
    documents: list[Document] = []

    summary_text = (
        f"Real Estate Dataset Summary:\n"
        f"Total Properties: {len(df_original)}\n"
        f"Price Range: ${df_original['price'].min()} to ${df_original['price'].max()}\n"
        f"Average Price: ${df_original['price'].mean():.2f}\n"
        f"Average Square Footage: {df_original['sqft_living'].mean():.2f} sq ft\n"
        f"Features tracked: {', '.join(df_original.columns)}\n"
    )
    documents.append(Document(page_content=summary_text, metadata={"document_type": "summary", "source": "dataset_summary"}))

    for idx, row in df_text.iterrows():
        property_text = (
            f"Property ID: {idx}\n"
            f"Price: ${row.get('price', 'N/A')}\n"
            f"Bedrooms: {row.get('bedrooms', 'N/A')}\n"
            f"Bathrooms: {row.get('bathrooms', 'N/A')}\n"
            f"Square Footage (Living): {row.get('sqft_living', 'N/A')} sq ft\n"
            f"Square Footage (Lot): {row.get('sqft_lot', 'N/A')} sq ft\n"
            f"Floors: {row.get('floors', 'N/A')}\n"
            f"Waterfront: {'Yes' if row.get('waterfront', 0) == 1 else 'No'}\n"
            f"View Quality: {row.get('view', 'N/A')} out of 5\n"
            f"Condition: {row.get('condition', 'N/A')} out of 5\n"
            f"Year Built: {row.get('yr_built', 'N/A')}\n"
            f"Year Renovated: {row.get('yr_renovated', 'N/A') if row.get('yr_renovated', 0) > 0 else 'Not renovated'}\n"
        )
        for col in ("zipcode", "city", "street", "neighborhood"):
            if col in row:
                property_text += f"{col.capitalize()}: {row[col]}\n"

        metadata = {
            "document_type": "property",
            "property_id": idx,
            "price": float(df_original.iloc[idx]["price"]) if "price" in df_original.columns else None,
            "bedrooms": float(df_original.iloc[idx]["bedrooms"]) if "bedrooms" in df_original.columns else None,
            "bathrooms": float(df_original.iloc[idx]["bathrooms"]) if "bathrooms" in df_original.columns else None,
            "sqft_living": float(df_original.iloc[idx]["sqft_living"]) if "sqft_living" in df_original.columns else None,
        }
        documents.append(Document(page_content=property_text, metadata=metadata))

    return documents


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_preprocess_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[float]]:
    """Load and preprocess the housing dataset."""
    dataset_path = str(settings.DATASET_PATH)
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return None, None, None

    try:
        file_mtime = os.path.getmtime(dataset_path)
        df = pd.read_csv(dataset_path)

        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
        df = df.replace(["NA", "N/A", "None", "NULL", ""], np.nan)
        df = df.dropna(subset=["price"])
        df = df[df["price"] != 0].reset_index(drop=True)

        numeric_cols = [
            "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
            "floors", "waterfront", "view", "condition", "sqft_above",
            "sqft_basement", "yr_built", "yr_renovated",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Information not available")
            else:
                df[col] = df[col].fillna(-1)

        df_for_text = df.copy()
        for col in df_for_text.columns:
            if df_for_text[col].dtype != "object":
                df_for_text[col] = df_for_text[col].apply(lambda x: "Information not available" if x == -1 else x)

        if "property_id" not in df.columns:
            df["property_id"] = df.index

        return df, df_for_text, file_mtime

    except Exception as exc:
        print(f"Error loading or preprocessing data: {exc}")
        return None, None, None


def setup_vector_db(
    df: pd.DataFrame,
    df_for_text: pd.DataFrame,
    file_mtime: float,
) -> Tuple[Optional[pd.DataFrame], Optional[Any]]:
    """Create or load the FAISS vector database."""
    if df is None:
        return None, None

    db_hash = _get_db_hash(file_mtime)
    db_path = str(settings.VECTOR_DB_DIR / f"faiss_index_{db_hash}")

    if os.path.exists(db_path):
        print(f"Loading existing vector database from {db_path}")
        try:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            print("Vector database loaded successfully!")
            return df, vector_store
        except Exception as exc:
            print(f"Error loading vector database: {exc}. Rebuilding…")

    print("Creating new vector database…")
    documents = _create_documents(df_for_text, df)

    property_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    summary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    texts: list[Document] = []
    for doc in documents:
        splitter = summary_splitter if doc.metadata.get("document_type") == "summary" else property_splitter
        texts.extend(splitter.split_documents([doc]))

    print(f"Total chunks to process: {len(texts)}")
    embeddings = OpenAIEmbeddings()
    batch_size = settings.VECTOR_BATCH_SIZE
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    vector_store = None
    start = time.time()
    for i, batch in enumerate(batches):
        if i == 0:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)
        processed = (i + 1) * batch_size
        pct = min(processed / len(texts) * 100, 100)
        print(f"Progress: {min(processed, len(texts))}/{len(texts)} ({pct:.1f}%) — batch {i+1}/{len(batches)}")

    print(f"Vector DB created in {time.time() - start:.1f}s")
    vector_store.save_local(db_path)
    return df, vector_store


def save_user_requirements(state: Dict[str, Any]) -> bool:
    """Persist user requirements and contact info to CSV."""
    contact_info = convert_numpy_types(state.get("contact_info", {}))
    requirements = convert_numpy_types(state.get("user_requirements", {}))

    if not contact_info.get("email") and not contact_info.get("whatsapp"):
        print("No contact information provided, skipping save")
        return False

    if not requirements:
        print("No requirements collected, skipping save")
        return False

    user_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": str(uuid.uuid4())[:8],
        "email": contact_info.get("email", ""),
        "whatsapp": contact_info.get("whatsapp", ""),
        "price_min": requirements.get("budget", ""),
        "bedrooms": requirements.get("bedrooms", ""),
        "bathrooms": requirements.get("bathrooms", ""),
        "sqft_living": requirements.get("sqft_living", ""),
        "location": requirements.get("location", ""),
        "property_type": requirements.get("property_type", ""),
        "specific_requirements": requirements.get("special_requirements", ""),
    }

    try:
        req_file = str(settings.USER_REQUIREMENTS_FILE)
        df_row = pd.DataFrame([convert_numpy_types(user_data)])
        file_exists = os.path.isfile(req_file)

        if file_exists:
            try:
                existing_df = pd.read_csv(req_file)
                if list(existing_df.columns) != list(df_row.columns):
                    df_row.to_csv(req_file, mode="w", header=True, index=False)
                else:
                    df_row.to_csv(req_file, mode="a", header=False, index=False)
            except Exception:
                df_row.to_csv(req_file, mode="w", header=True, index=False)
        else:
            df_row.to_csv(req_file, mode="w", header=True, index=False)

        print(f"User requirements saved to {req_file}")
        state["requirements_saved"] = True
        return True

    except Exception as exc:
        print(f"Error saving user requirements: {exc}")
        return False
