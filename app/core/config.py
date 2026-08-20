import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# BASE_DIR points to the project root (where .env and json files should live)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # --- Project Metadata ---
    PROJECT_NAME: str = "Brain Tumor Project"
    
    # --- API Keys (Pydantic automatically looks for these names in .env) ---
    COHERE_API_KEY: str
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str
    SERP_API_KEY: str
    GROQ_API_KEY: str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    
    # --- Database Configuration ---
    # Automatically falls back to local if not provided (useful for AWS RDS later)
    DATABASE_URL: str

    LANGGRAPH_DB_URL:str
    
    # --- File Paths (Dynamic for Local/Cloud compatibility) ---
    # We look for the JSON file in the project root
    

    # --- Pydantic Config ---
    # This tells Pydantic where to find the .env file
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore" # Ignores extra variables in .env that aren't defined here
    )

# Create a singleton instance to be used across the app
settings = Settings()

# On HF Spaces: write JSON content to temp file
