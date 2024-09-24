import os
from dotenv import load_dotenv
from openai import OpenAI
from litellm import completion as LiteLLMCompletion
from langfuse import Langfuse

load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Initialize Langfuse client with environment variables
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SK"),
    public_key=os.getenv("LANGFUSE_PK"),
    host=os.getenv("LANGFUSE_HOST"),
)

__all__ = ['client', 'langfuse']
