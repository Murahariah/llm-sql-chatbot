# path: llm-sql-chatbot/app.py

import streamlit as st
import database as db
import llm_handler as llm
from streamlit_chat import message
import base64

# --- Page Configuration ---
st.set_page_config(page_title="Support Chatbot", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .main {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .stTextInput>div>input {border: 1px solid #ccc; border-radius: 5px; padding: 10px;}
    .stButton>button {background-color: #0078d4; color: white; border-radius: 5px; padding: 8px 16px;}
    .stSpinner {margin: 0 auto; display: block;}
    .chat-container {max-height: 500px; overflow-y: auto; padding: 10px; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
    .icon {vertical-align: middle; margin-right: 5px;}
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("Customer Support Chatbot 🤖")
st.subheader("Your AI Assistant for Quick Answers 🚀")
try:
    with open("static/chatbot.png", "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
    st.markdown(f'<img src="data:image/png;base64,{img_data}" width="50" style="vertical-align: middle; margin-right: 10px;">', unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown('<span class="icon">🤖</span>', unsafe_allow_html=True)
st.markdown("---")

# --- Initialize Database ---
@st.cache_resource
def initialize_db():
    try:
        db.init_db()
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False

if not initialize_db():
    st.stop()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]
if "context_history" not in st.session_state:
    st.session_state.context_history = []

# --- Chat Interface ---
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=(msg["role"] == "user"), key=f"msg_{i}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- User Input ---
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        user_query = st.text_input("Ask me anything: ✍️", placeholder="E.g., What's the status of ORD1234?", key="input")
    with col2:
        submit_button = st.form_submit_button(label="Send 📤")
    with col3:
        clear_chat = st.form_submit_button(label="🗑️ Clear Chat")

if clear_chat:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]
    st.session_state.context_history = []
    st.success("Chat history cleared! ✅")

# --- Suggested Questions ---
st.markdown("**Quick Questions:**")
suggestions = [
    ("What's the price of the iPhone 15?", "💰"),
    ("How do I track my order ORD5678?", "📦"),
    ("What’s the support number for billing?", "📞"),
    ("How do I reset my password?", "🔑")
]
cols = st.columns(4)
for i, (suggestion, icon) in enumerate(suggestions):
    with cols[i]:
        if st.button(f"{icon} {suggestion}", key=f"suggest_{i}"):
            user_query = suggestion
            submit_button = True

# --- Processing Logic ---
if submit_button and user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Thinking... 💭"):
        query_lower = user_query.lower()
        retrieved_context = []

        order_id = db.find_order_id(user_query)
        if order_id:
            retrieved_context = db.fetch_order_status(order_id)
        elif "order" in query_lower and ("status" in query_lower or "track" in query_lower):
            retrieved_context = [{"info": "Please provide your Order ID (e.g., ORD1234)."}]
        elif any(kw in query_lower for kw in ["price", "cost", "feature", "product", "buy"]):
            known_products = {"galaxy s23": "Galaxy S23", "pixel 8 pro": "Pixel 8 Pro", "iphone 15": "iPhone 15", "zenbook": "Zenbook Laptop"}
            potential_product = next((prod for key, prod in known_products.items() if key in query_lower), None)
            retrieved_context = db.fetch_product_info(potential_product) if potential_product else [{"info": "Which product are you asking about?"}]
        elif any(kw in query_lower for kw in ["support", "contact", "phone", "email", "call"]):
            department = {"technical": "Technical Support", "sales": "Sales", "billing": "Billing"}.get(next((kw for kw in ["technical", "sales", "billing"] if kw in query_lower), None))
            retrieved_context = db.fetch_support_contacts(department)
        else:
            retrieved_context = db.fetch_faq(user_query)

        final_context = retrieved_context if retrieved_context else [{"info": "No specific information found."}]
        response = llm.get_rag_response(user_query, final_context, st.session_state.context_history)

        st.session_state.context_history.append({"query": user_query, "context": final_context, "response": response})
        if len(st.session_state.context_history) > 5:
            st.session_state.context_history = st.session_state.context_history[-5:]

    st.session_state.messages.append({"role": "assistant", "content": response})
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, msg in enumerate(st.session_state.messages):
            message(msg["content"], is_user=(msg["role"] == "user"), key=f"msg_{i}_rerender")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Powered by Murahari ⚡ • Built with Streamlit 🌟 • April 2025")