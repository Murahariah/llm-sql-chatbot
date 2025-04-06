# path: llm-sql-chatbot/llm_handler.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from cachetools import TTLCache

# --- LLM Setup ---
try:
    llm = ChatOllama(model="mistral")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

output_parser = StrOutputParser()
cache = TTLCache(maxsize=100, ttl=3600)  # Cache 100 responses for 1 hour

# --- Prompt Template ---
rag_prompt_template = """
You are a customer support assistant. Answer the user's question using only the provided context, rephrasing raw data into concise, conversational language. Use conversation history for follow-ups if relevant. If no answer is found, say so politely.

Conversation History:
{history}

Current Context:
{context}

User Query:
{query}

Response:
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

# --- Langchain Chain ---
if llm:
    rag_chain = rag_prompt | llm | output_parser
else:
    rag_chain = None

# --- Function to get Response ---
def get_rag_response(user_query: str, retrieved_context: list[dict], context_history: list[dict]):
    if not rag_chain:
        return "Error: LLM unavailable."

    # Cache key: query + simplified context
    cache_key = f"{user_query}::{str(retrieved_context)[:100]}"
    if cache_key in cache:
        return cache[cache_key]

    history_str = "\n".join([f"Q: {item['query']} A: {item['response']}" for item in context_history[-3:]]) if context_history else "No history."
    context_str = "\n".join([str(item) for item in retrieved_context]) if retrieved_context else "No info found."

    try:
        response = rag_chain.invoke({"history": history_str, "context": context_str, "query": user_query})
        cache[cache_key] = response
        return response
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return "Sorry, an error occurred."