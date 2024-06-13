from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Access the API key using the variable name defined in the .env file
api_key = os.getenv("OPENAI_API_KEY")

def get_embedding_function():

    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
    return embeddings