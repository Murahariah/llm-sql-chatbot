# LLM-SQL-Chatbot

A customer support chatbot powered by an LLM (Mistral via Ollama), SQLite database, and Streamlit. It provides natural, conversational responses to queries about product information, order status, support contacts, and FAQs, with support for follow-up questions using session memory.

## Features

- **Natural Language Responses**: Rephrases raw database output into friendly, conversational text.
- **Intent Recognition**: Detects user intent (e.g., product price, order status) and fetches relevant data from SQLite.
- **Session Memory**: Remembers previous queries and responses within a session to handle follow-up questions (e.g., "Tell me more about that").
- **Retry Logic**: Automatically retries failed SQL queries up to 3 times with exponential backoff for robustness.
- **Fallback Plan**: Gracefully handles missing data or errors with polite responses.
- **Interactive UI**: Built with Streamlit, featuring a chat interface, suggested questions with icons, and a "Clear Chat" option.
- **Database Integration**: Uses SQLite to store and retrieve product info, order status, support contacts, and FAQs.
- **Quick Questions**: Predefined buttons for common queries to enhance user experience.

## How LLM is Used

The chatbot leverages the Mistral model (via Ollama) for natural language processing within a Retrieval-Augmented Generation (RAG) framework. Here’s how it works:

### Prompt Example
The LLM is guided by a prompt that includes conversation history, current context, and the user query. Example:

```
Conversation History (previous queries, contexts, and responses):
Query: What’s the price of the iPhone 15?
Context: {'name': 'iPhone 15', 'features': 'A16 Bionic chip, Dynamic Island, USB-C', 'price': 849.0}
Response: The iPhone 15 costs $849.

Current Context (raw database output):
{'name': 'iPhone 15', 'features': 'A16 Bionic chip, Dynamic Island, USB-C', 'price': 849.0}

User Query:
What are its features?

Assistant Response:
The iPhone 15 comes with an A16 Bionic chip, Dynamic Island, and a USB-C port.
```

- **Instructions**: The LLM is told to rephrase raw data naturally, use history for follow-ups, and avoid speculation beyond provided data.

### Retry Logic
- **Where**: Implemented in `database.py`’s `_execute_query` function using the `tenacity` library.
- **Details**:
  - Retries up to 3 times on `sqlite3.Error` (e.g., database locked).
  - Uses exponential backoff (1s, 2s, 4s, max 10s) between attempts.
  - Logs retry attempts for debugging (e.g., "Retrying SQL query (attempt 2)...").
- **Purpose**: Ensures robustness against transient database issues.

### Fallback Plan
- **Empty Context**: If no data is retrieved, the LLM receives: "No specific information found in the database for this query," prompting a response like, "I don’t have the details to answer that right now."
- **LLM Unavailable**: Returns "Error: LLM is not available. Please ensure Ollama is running with the Mistral model."
- **Unexpected Errors**: Catches exceptions and returns "Sorry, I encountered an error while processing your request."

## Prerequisites

- **Python**: 3.8 or higher.
- **Ollama**: For running the Mistral LLM locally.
- **Dependencies**: Listed in `requirements.txt`.

## Project Structure

```
llm-sql-chatbot/
├── data/
│   └── chatbot.db        # SQLite database (auto-created)
├── app.py                # Main Streamlit application
├── database.py           # Database setup and query functions
├── llm_handler.py        # LLM interaction logic
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## How to Download and Use the Ollama Model

1. **Install Ollama**:
   - Download and install Ollama from [ollama.ai](https://ollama.ai/).
   - For Windows/Linux/macOS, follow the platform-specific instructions.

2. **Pull the Mistral Model**:
   - Open a terminal and run:
     ```
     ollama pull mistral
     ```
   - This downloads the Mistral model locally (several GB, depending on the version).

3. **Run Ollama**:
   - Start the Ollama server:
     ```
     ollama run mistral
     ```
   - Keep this terminal open while using the chatbot. Alternatively, run it in the background (e.g., `ollama run mistral &` on Linux).

## Step-by-Step Instructions to Run the Code

1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd llm-sql-chatbot
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   - Ensure `requirements.txt` contains:
     ```
      streamlit==1.44.1
      langchain==0.3.23
      langchain-community==0.3.21 
      streamlit-chat==0.1.1
      tenacity==9.1.2
      cachetools==5.5.2
     ```
   - Run:
     ```
     pip install -r requirements.txt
     ```

4. **Start the Ollama Server**:
   - In a separate terminal:
     ```
     ollama run mistral
     ```

5. **Run the Application**:
   - In the project directory:
     ```
     streamlit run app.py
     ```
   - Open your browser to `http://localhost:8501`.

6. **Interact with the Chatbot**:
   - Type a query (e.g., "What’s the price of the iPhone 15?") or click a suggested question.
   - Follow up with related questions (e.g., "What are its features?").
   - Use the "Clear Chat" button to reset the conversation.

## Troubleshooting

- **LLM Error**: Ensure Ollama is running and the Mistral model is pulled (`ollama list` to check).
- **Database Error**: Delete `data/chatbot.db` and rerun to recreate it if corrupted.
- **Port Conflict**: If `8501` is in use, Streamlit will prompt an alternative port.

