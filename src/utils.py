import os
from dotenv import load_dotenv

def load_env(dotenv_path=".env"):
    """
    Load environment variables from a .env file if it exists.
    Defaults to `.env` in project root.
    """
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=True)
        print(f"✅ Loaded environment variables from {dotenv_path}")
    else:
        print("ℹ️ No .env file found. Make sure API keys are set in your environment.")
