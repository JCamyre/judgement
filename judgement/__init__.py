import os
import litellm
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse
from together import Together

load_dotenv()

# Set callbacks
litellm.success_callback = ["langfuse"]  # log input/output to langfuse
# Initialize OpenAI client
client = OpenAI()

# Initialize Langfuse client with environment variables
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

together_client = Together(
    api_key=os.getenv("TOGETHERAI_API_KEY"),
)

__all__ = ['client', 'langfuse', 'together_client']
