# agents/text_agent.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from state import AgentState
from .models import llm_gemini_pro

def text_analyst_node(state: AgentState) -> dict:
    """ 🔎 The Text Analyst (using Gemini) """
    print("---🔎 EXECUTING TEXT ANALYST (using Gemini Pro)---")
    loader = UnstructuredFileLoader(state["document_path"])
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    search_query_gen = llm_gemini_pro.invoke(f"Generate a concise search query for: '{state['query']}'")
    search_query = search_query_gen.content
    retrieved_docs = retriever.invoke(search_query)
    relevant_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    return {"agent_outputs": [f"### Text Analyst Findings:\n\n{relevant_text}"]}