# agents/models.py

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# The model best for vision and general reasoning
llm_openai_vision = ChatOpenAI(model="gpt-4o", temperature=0)

# The model best for large context summarization and text tasks
llm_gemini_pro = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

# A fast and cheap model specifically for the routing task
llm_router = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)