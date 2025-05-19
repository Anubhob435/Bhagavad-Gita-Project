import os
import time
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logger = logging.getLogger(__name__)

def get_gemini_llm():
    """
    Initialize and return a Gemini language model.
    First tries Gemini 2.0 Flash, then falls back to Gemini 1.5 Pro if rate limited.
    API key is loaded from environment variables.
    Will try with alternate API key if primary key fails.
    """
    # Load environment variables
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    alternate_key = os.getenv("ALTERNATE_GEMINI_API_KEY")
    
    if not gemini_api_key:
        error_msg = "GEMINI_API_KEY not found in environment variables"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Configure retry settings - these can help with rate limits
    retry_models = [
        "gemini-2.0-flash", # Try this model first (higher quota)
        "gemini-2.5-flash"         # Fall back to this older model if needed
    ]
    
    # Try each available key with each model until one works
    for api_key in [gemini_api_key, alternate_key]:
        if not api_key:
            continue
            
        for model in retry_models:
            try:
                logger.info(f"Trying to initialize {model} with API key")
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=0.2,  # Slightly higher temperature for more varied responses
                    google_api_key=api_key,
                    convert_system_message_to_human=True,  # Better compatibility across models
                    max_retries=1  # Only retry once per request to avoid quota issues
                )
                # Test the model with a simple query to make sure it works
                llm.invoke("Hello")
                logger.info(f"Successfully initialized {model}")
                return llm
            except Exception as e:
                logger.warning(f"Error with {model}: {e}")
                time.sleep(1)  # Add a small delay before trying the next option
    
    # If we get here, all options failed
    logger.error("All Gemini models and API keys failed")
    raise ValueError("Unable to initialize any Gemini model with available API keys")