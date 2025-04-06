
# LLM-Powered Chatbot with SQL Integration

## Objective

This project implements a chatbot powered by a Large Language Model (Mistral via Ollama) that interacts with a local SQLite database. It understands user queries in natural language, retrieves relevant information from the database, and provides conversational responses.

## Features

* **Natural Language Understanding:** Takes user questions via a Streamlit web interface.
* **Intent Recognition:** Uses basic keyword matching to identify the user's intent (e.g., checking order status, finding product info, getting support contacts, asking FAQs).
* **Retrieval-Augmented Generation (RAG):**
    * Identifies the relevant SQL table based on intent.
    * Fetches relevant rows from the SQLite database to use as context.
    * Does **not** generate SQL queries directly using the LLM for safety and simplicity.
* **LLM Interaction:** Uses Langchain and the Mistral model (via Ollama) to generate responses based *only* on the retrieved database context.
* **Conversational Responses:** Provides informative answers in a natural way.
* **Database:** Uses SQLite with predefined tables (`product_info`, `order_status`, `support_contacts`, `faq`).

## Tech Stack

* **LLM:** Mistral (running locally via Ollama)
* **LLM Orchestration:** Langchain (`langchain`, `langchain-community`)
* **Backend/Logic:** Python 3
* **Database:** SQLite 3
* **UI:** Streamlit

## Project Structure

Okay, let's design and build the LLM-powered chatbot with SQL integration using the specified stack. We'll focus on the Retrieval-Augmented Generation (RAG) approach (Option B) for robustness and safety.

Core Concepts:

Intent Recognition: We'll use simple keyword matching initially to determine the user's likely intent and the relevant SQL table. A more advanced approach could use the LLM itself for intent classification.
Data Retrieval (RAG): Based on the intent, we'll query the relevant SQLite table to fetch potential context.
LLM Interaction (Langchain + Mistral): We'll use Langchain to structure the interaction. The retrieved SQL data will be formatted and sent to the Mistral LLM along with the user's query as context.
Prompt Engineering: A specific prompt will instruct the LLM to answer the user's question based only on the provided context (the retrieved SQL data).
Response Generation: The LLM generates a conversational response based on the context.
UI (Streamlit): A simple web interface for user interaction.
Project Structure:

llm-sql-chatbot/
├── data/
│   └── chatbot.db        # SQLite database file (will be created)
├── app.py                # Main Streamlit application
├── database.py           # Database setup and query functions
├── llm_handler.py        # LLM interaction logic (Langchain, Mistral)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
Step-by-Step Implementation:

1. database.py - Database Setup and Retrieval

Purpose: Handles all SQLite database operations: initialization, table creation, sample data insertion, and data fetching for RAG.
Details:
Uses the sqlite3 standard library.
Defines functions to create tables (product_info, order_status, support_contacts, faq) if they don't exist.
Includes functions to insert sample data (using INSERT OR IGNORE to prevent duplicates on rerun).
Provides specific retrieval functions (Workspace_*) for each table based on simple keyword matching or specific IDs. These functions will be called by app.py.
Python

# path: llm-sql-chatbot/database.py

import sqlite3
import os
import re

# --- Database Setup ---
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "chatbot.db")

def init_db():
    """Initializes the database, creates tables, and inserts sample data."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS product_info (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            features TEXT,
            price REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS order_status (
            order_id TEXT PRIMARY KEY,
            customer_name TEXT,
            status TEXT NOT NULL CHECK(status IN ('Processing', 'Shipped', 'Delivered', 'Cancelled'))
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS support_contacts (
            contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            department TEXT UNIQUE NOT NULL,
            phone TEXT,
            email TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faq (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE NOT NULL,
            keywords TEXT, -- Comma-separated keywords for simple matching
            answer TEXT NOT NULL
        )
    ''')

    # Insert Sample Data (Ignore if already exists)
    sample_data = [
        ("product_info", "INSERT OR IGNORE INTO product_info (name, features, price) VALUES (?, ?, ?)",
         [('Galaxy S23', 'Latest Samsung flagship phone, AI features, great camera', 799.99),
          ('Pixel 8 Pro', 'Google Tensor G3, advanced camera system, AI magic', 999.00),
          ('iPhone 15', 'A16 Bionic chip, Dynamic Island, USB-C', 849.00),
          ('Zenbook Laptop', 'OLED display, Intel Core i7, 16GB RAM', 1199.50)]),

        ("order_status", "INSERT OR IGNORE INTO order_status (order_id, customer_name, status) VALUES (?, ?, ?)",
         [('ORD5678', 'Alice Smith', 'Shipped'),
          ('ORD1234', 'Bob Johnson', 'Processing'),
          ('ORD9901', 'Charlie Brown', 'Delivered'),
          ('ORD7755', 'Diana Prince', 'Cancelled')]),

        ("support_contacts", "INSERT OR IGNORE INTO support_contacts (department, phone, email) VALUES (?, ?, ?)",
         [('Sales', '1-800-123-4567', 'sales@example.com'),
          ('Technical Support', '1-888-TECH-HLP', 'support@example.com'),
          ('Billing', '1-800-BILLING', 'billing@example.com')]),

        ("faq", "INSERT OR IGNORE INTO faq (question, keywords, answer) VALUES (?, ?, ?)",
         [('How do I reset my password?', 'password,reset,account,login', 'You can reset your password by clicking the "Forgot Password" link on the login page and following the instructions sent to your email.'),
          ('What is the return policy?', 'return,policy,refund,exchange', 'Our return policy allows returns within 30 days of purchase with the original receipt. Please see our website for full details.'),
          ('How to track my order?', 'track,order,shipping,status', 'You can track your order using the tracking link sent to your email after shipping, or by entering your order ID on our website.')])
    ]

    for table, query, data in sample_data:
        cursor.executemany(query, data)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# --- Data Retrieval Functions ---

def _execute_query(query, params=()):
    """Helper function to execute a SELECT query and fetch results."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row # Return rows as dict-like objects
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        # Convert Row objects to plain dictionaries for easier handling
        return [dict(row) for row in results]
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return [] # Return empty list on error

def find_order_id(text):
    """Extracts a potential order ID (e.g., ORDxxxx) from text."""
    match = re.search(r'\b(ORD\d{4,})\b', text, re.IGNORECASE)
    return match.group(1) if match else None

def fetch_order_status(order_id):
    """Fetches status for a specific order ID."""
    query = "SELECT order_id, customer_name, status FROM order_status WHERE order_id = ?"
    return _execute_query(query, (order_id,))

def fetch_product_info(product_name_keyword):
    """Fetches product info matching a keyword."""
    # Prioritize exact match first
    query_exact = "SELECT name, features, price FROM product_info WHERE name LIKE ?"
    results_exact = _execute_query(query_exact, (product_name_keyword,))
    if results_exact:
        return results_exact

    # Fallback to partial match if no exact match
    query_partial = "SELECT name, features, price FROM product_info WHERE name LIKE ?"
    return _execute_query(query_partial, (f'%{product_name_keyword}%',))


def fetch_support_contacts(department_keyword=None):
    """Fetches support contacts, optionally filtered by department keyword."""
    if department_keyword:
        query = "SELECT department, phone, email FROM support_contacts WHERE department LIKE ?"
        return _execute_query(query, (f'%{department_keyword}%',))
    else:
        # Fetch all contacts if no specific department mentioned
        query = "SELECT department, phone, email FROM support_contacts"
        return _execute_query(query)

def fetch_faq(query_text):
    """Fetches FAQ entries matching keywords in the question."""
    # Simple keyword matching (split query and check against keywords/question)
    # More advanced: Use embeddings or full-text search if available
    words = query_text.lower().split()
    if not words:
        return []

    # Build a query with multiple LIKE clauses
    conditions = " OR ".join(["question LIKE ? OR keywords LIKE ?"] * len(words))
    query = f"SELECT question, answer FROM faq WHERE {conditions}"

    # Create parameters for each word, checking both question and keywords
    params = []
    for word in words:
        like_pattern = f'%{word}%'
        params.extend([like_pattern, like_pattern])

    # Limit results to avoid overwhelming the context
    query += " LIMIT 5"

    return _execute_query(query, tuple(params))

if __name__ == '__main__':
    # Example usage when running the script directly
    init_db()
    print("\nSample Queries:")
    print("Order ORD5678:", fetch_order_status('ORD5678'))
    print("Product 'Galaxy':", fetch_product_info('Galaxy S23'))
    print("Product 'pixel':", fetch_product_info('pixel')) # partial match
    print("Support 'Tech':", fetch_support_contacts('Technical'))
    print("All Support:", fetch_support_contacts())
    print("FAQ 'password':", fetch_faq('reset password'))
2. llm_handler.py - LLM Interaction

Purpose: Manages communication with the Mistral LLM using Langchain.
Details:
Uses ChatOllama from langchain_community.chat_models to connect to a locally running Mistral model via Ollama. Make sure Ollama is installed and the Mistral model is pulled (ollama pull mistral).
Defines a ChatPromptTemplate specifically for the RAG task. It instructs the LLM to use only the provided database context.
Creates a Langchain "chain" (prompt | llm | output_parser) to process requests.
Provides a function get_rag_response that takes the user query and retrieved context, formats them into the prompt, and returns the LLM's generated response.
Python

# path: llm-sql-chatbot/llm_handler.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama # Use ChatOllama for local models

# --- LLM Setup ---
# Ensure Ollama is running and the 'mistral' model is available
# Run `ollama pull mistral` in your terminal if needed.
try:
    llm = ChatOllama(model="mistral") # Specify the local model name
except Exception as e:
    print(f"Error initializing LLM. Make sure Ollama is running and the 'mistral' model is available. Error: {e}")
    # Fallback or exit if LLM is critical
    llm = None # Indicate LLM is not available

output_parser = StrOutputParser()

# --- Prompt Template for RAG ---
# Instructs the LLM how to behave based on retrieved context
rag_prompt_template = """
You are a helpful customer support assistant. Your task is to answer the user's question based *only* on the context provided below.

If the context contains the information needed to answer the question, formulate a concise and helpful response based *directly* on that context.
If the context *does not* contain relevant information to answer the question, clearly state that you don't have the specific information based on the provided details.
Do not make up information or answer based on your general knowledge.

Context:
{context}

User Query:
{query}

Assistant Response:
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

# --- Langchain Chain ---
if llm:
    rag_chain = rag_prompt | llm | output_parser
else:
    rag_chain = None # Chain is not usable without LLM

# --- Function to get Response ---
def get_rag_response(user_query: str, retrieved_context: list[dict]):
    """
    Generates a response using the LLM based on the user query and retrieved context.

    Args:
        user_query: The question asked by the user.
        retrieved_context: A list of dictionaries representing data fetched from the DB.

    Returns:
        A string containing the LLM's response, or an error message.
    """
    if not rag_chain:
        return "Error: LLM is not available. Please ensure Ollama is running with the Mistral model."

    if not retrieved_context:
        # Handle cases where the initial DB query found nothing relevant
        # Option 1: Try a fallback (like general FAQ search) - not implemented here yet
        # Option 2: Inform the LLM the context is empty
        context_str = "No specific information found in the database for this query."
    else:
        # Format the retrieved data into a readable string for the LLM
        context_str = "\n---\n".join([str(item) for item in retrieved_context])

    try:
        response = rag_chain.invoke({"context": context_str, "query": user_query})
        return response
    except Exception as e:
        print(f"Error invoking LLM chain: {e}")
        return "Sorry, I encountered an error while processing your request."

3. app.py - Streamlit UI and Orchestration

Purpose: Provides the user interface and coordinates the workflow: gets user input, determines intent, calls database functions, calls LLM handler, and displays the result.
Details:
Imports streamlit, database, and llm_handler.
Initializes the database on startup using database.init_db().
Creates a title and a text input field for the user query.
Intent Logic: When input is received:
Uses simple if/elif/else based on keywords (order, status, price, product, support, contact, password, how to, etc.) to decide which database function to call.
Extracts necessary entities (like Order ID using database.find_order_id or product names).
Database Call: Calls the appropriate Workspace_* function from database.py.
LLM Call: Passes the user's original query and the fetched database results (context) to llm_handler.get_rag_response.
Display: Shows the final, conversational response generated by the LLM using st.write or st.chat_message. Includes basic handling for when no context is found or LLM fails.
Python

# path: llm-sql-chatbot/app.py

import streamlit as st
import database as db
import llm_handler as llm

# --- Page Configuration ---
st.set_page_config(page_title="Support Chatbot", layout="centered")
st.title("LLM-Powered Support Chatbot 🤖")

# --- Initialize Database ---
# This runs once when the script starts
try:
    db.init_db()
except Exception as e:
    st.error(f"Failed to initialize the database: {e}")
    st.stop() # Stop the app if DB init fails

# --- Chat Interface ---
st.write("Ask me about product prices, order status, support contacts, or FAQs!")

# Use session state to potentially store chat history later
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages (optional, good for context)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_query = st.chat_input("What can I help you with?")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # --- Intent Recognition and Data Retrieval ---
    retrieved_context = []
    query_lower = user_query.lower()

    # 1. Check for Order Status intent
    order_id = db.find_order_id(user_query)
    if order_id:
        st.write(f"Looking up order status for {order_id}...")
        retrieved_context = db.fetch_order_status(order_id)
    elif "order" in query_lower and ("status" in query_lower or "track" in query_lower):
         # Ask for order ID if keywords match but ID is missing
         retrieved_context = [{"info": "Please provide the specific Order ID (e.g., ORD1234) you want to check."}]


    # 2. Check for Product Info intent (price, features)
    # Simple entity extraction: look for keywords and potential product names
    # This is basic; needs improvement for real-world use (e.g., NER)
    elif "price" in query_lower or "cost" in query_lower or "feature" in query_lower or "product" in query_lower or "buy" in query_lower:
        # Try to extract a potential product name (simple approach)
        potential_product = ""
        # A list of known product keywords might help here
        known_products = ["Galaxy S23", "Pixel 8 Pro", "iPhone 15", "Zenbook Laptop"] # Can be loaded dynamically
        for prod in known_products:
            if prod.lower() in query_lower:
                potential_product = prod
                break
        # If no known product found, try extracting capitalized words or nouns (more complex)
        # For now, if a likely keyword is found, fetch matching products
        if potential_product:
             st.write(f"Looking up product info for '{potential_product}'...")
             retrieved_context = db.fetch_product_info(potential_product)
        else:
             # If keywords like 'price' are present but no specific product identified
             retrieved_context = [{"info": "Which product are you asking about?"}]


    # 3. Check for Support Contact intent
    elif "support" in query_lower or "contact" in query_lower or "phone" in query_lower or "email" in query_lower or "call" in query_lower:
        department = None
        # Simple check for department names
        if "technical" in query_lower: department = "Technical Support"
        elif "sales" in query_lower: department = "Sales"
        elif "billing" in query_lower: department = "Billing"

        st.write(f"Looking up contact info {f'for {department}' if department else ''}...")
        retrieved_context = db.fetch_support_contacts(department) # Pass None if no specific dept found


    # 4. Check for FAQ intent (fallback)
    else:
        st.write("Checking FAQs...")
        retrieved_context = db.fetch_faq(user_query)


    # --- Generate Response using LLM with RAG ---
    if not retrieved_context:
        # If no specific context was retrieved by intent matching, provide a generic non-committal context for the LLM
         final_context = [{"info": "No specific information found for the query in the knowledge base."}]
         # Alternative: Directly give a fallback message without calling LLM
         # response = "I couldn't find specific information for your query. Could you please rephrase or ask something else?"
    else:
         final_context = retrieved_context

    st.write("Generating response...") # Indicate LLM processing
    response = llm.get_rag_response(user_query, final_context)


    # --- Display Assistant Response ---
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

4. requirements.txt - Dependencies

Purpose: Lists all necessary Python packages for installation.
Details: Includes Streamlit for the UI, Langchain core and community components (for Ollama integration).
Plaintext

# path: llm-sql-chatbot/requirements.txt

streamlit
langchain>=0.1.0,<0.2.0
langchain-community>=0.0.20,<0.1.0 # Includes ChatOllama
# No need to list sqlite3 as it's built-in
5. README.md - Documentation

Purpose: Explains the project, setup, usage, and design choices.
Details: Provides clear instructions for users to set up and run the chatbot. Explains the RAG approach and how the LLM is utilized.
Markdown

# path: llm-sql-chatbot/README.md

# LLM-Powered Chatbot with SQL Integration

## Objective

This project implements a chatbot powered by a Large Language Model (Mistral via Ollama) that interacts with a local SQLite database. It understands user queries in natural language, retrieves relevant information from the database, and provides conversational responses.

## Features

* **Natural Language Understanding:** Takes user questions via a Streamlit web interface.
* **Intent Recognition:** Uses basic keyword matching to identify the user's intent (e.g., checking order status, finding product info, getting support contacts, asking FAQs).
* **Retrieval-Augmented Generation (RAG):**
    * Identifies the relevant SQL table based on intent.
    * Fetches relevant rows from the SQLite database to use as context.
    * Does **not** generate SQL queries directly using the LLM for safety and simplicity.
* **LLM Interaction:** Uses Langchain and the Mistral model (via Ollama) to generate responses based *only* on the retrieved database context.
* **Conversational Responses:** Provides informative answers in a natural way.
* **Database:** Uses SQLite with predefined tables (`product_info`, `order_status`, `support_contacts`, `faq`).

## Tech Stack

* **LLM:** Mistral (running locally via Ollama)
* **LLM Orchestration:** Langchain (`langchain`, `langchain-community`)
* **Backend/Logic:** Python 3
* **Database:** SQLite 3
* **UI:** Streamlit

## Project Structure

llm-sql-chatbot/
├── data/
│   └── chatbot.db        # SQLite database file (auto-created)
├── app.py                # Main Streamlit application
├── database.py           # Database setup and query functions
├── llm_handler.py        # LLM interaction logic (Langchain, Mistral/Ollama)
├── requirements.txt      # Python dependencies
└── README.md             # This file


## Setup and Running

1.  **Prerequisites:**
    * Python 3.8+ installed.
    * [Ollama](https://ollama.com/) installed and running.
    * Mistral model pulled via Ollama: `ollama pull mistral`

2.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd llm-sql-chatbot
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit App:**
    * Ensure Ollama is running in the background.
    * Execute the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
    This will start the database initialization (if needed) and open the chatbot interface in your web browser.

## How the LLM is Used (RAG Approach)

1.  **User Input:** The user types a question into the Streamlit interface.
2.  **Intent/Retrieval:** `app.py` performs basic keyword analysis on the input to guess the intent and identify relevant entities (like an order ID or product name). Based on this, it calls specific functions in `database.py` to query the SQLite database.
3.  **Context Formatting:** The data fetched from the database (e.g., order details, product info) is formatted into a text string. If no relevant data is found, a message indicating this is used as context.
4.  **LLM Prompting:** `llm_handler.py` uses a Langchain `ChatPromptTemplate`. This template explicitly instructs the Mistral LLM:
    * To act as a customer support assistant.
    * To answer the `User Query` **solely based on the provided `Context`** (the formatted database data).
    * To state if the context doesn't contain the answer, rather than hallucinating.
    * *Example Prompt Structure (Simplified):*
        ```
        System: You are a helpful assistant. Answer based ONLY on the context. If context is empty or irrelevant, say you don't know.
        Context: {'order_id': 'ORD5678', 'customer_name': 'Alice Smith', 'status': 'Shipped'}
        User Query: What is the status of my order ORD5678?
        Assistant:
        ```
5.  **Response Generation:** The constructed prompt (including context and user query) is sent to the Mistral LLM via the `ChatOllama` integration. The LLM generates the final conversational response.
6.  **Fallback Plan:** If the initial database retrieval yields no results (`retrieved_context` is empty), the LLM is either informed that no context was found or a predefined message is shown directly in `app.py`. If the LLM itself fails (e.g., Ollama not running), an error message is displayed.

## Limitations and Future Work

* **Intent Recognition:** Relies on simple keywords; could be improved using embeddings, classification models, or an LLM call dedicated to intent/entity extraction.
* **Entity Extraction:** Basic extraction for order IDs and product names; vulnerable to variations in user phrasing. Named Entity Recognition (NER) models could enhance this.
* **SQL Generation:** This implementation uses RAG. Option A (SQL Generation by LLM) was not implemented here but could be added (requires careful prompting with schema, validation, and security measures).
* **Error Handling:** Basic error handling is present, but could be more robust (e.g., specific database errors, LLM timeouts).
* **Session Memory:** Currently stateless between queries (though Streamlit `chat_input` provides some visual history). True conversational memory (using `st.session_state` to store history and potentially feed it back to the LLM) could handle follow-up questions better.
* **Deployment:** The app runs locally. It could be deployed using services like Streamlit Community Cloud, Hugging Face Spaces, or Render.
