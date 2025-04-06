# path: llm-sql-chatbot/database.py

import sqlite3
import os
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Database Setup ---
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "chatbot.db")

def init_db():
    """Initializes the database, creates tables with indexes, and inserts sample data."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Tables with Indexes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS product_info (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            features TEXT,
            price REAL
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_product_name ON product_info(name)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS order_status (
            order_id TEXT PRIMARY KEY,
            customer_name TEXT,
            status TEXT NOT NULL CHECK(status IN ('Processing', 'Shipped', 'Delivered', 'Cancelled'))
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_id ON order_status(order_id)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS support_contacts (
            contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            department TEXT UNIQUE NOT NULL,
            phone TEXT,
            email TEXT
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_department ON support_contacts(department)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faq (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE NOT NULL,
            keywords TEXT,
            answer TEXT NOT NULL
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_faq_keywords ON faq(keywords)')

    # Insert Sample Data
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
         [('How do I reset my password?', 'password,reset,account,login', 'You can reset your password by clicking "Forgot Password" on the login page.'),
          ('What is the return policy?', 'return,policy,refund,exchange', 'Returns are allowed within 30 days with the original receipt.'),
          ('How to track my order?', 'track,order,shipping,status', 'Track your order with the link emailed after shipping or on our website.')])
    ]

    for table, query, data in sample_data:
        cursor.executemany(query, data)

    conn.commit()
    conn.close()
    print("Database initialized with indexes.")

# --- Data Retrieval Functions ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(sqlite3.Error))
def _execute_query(query, params=()):
    """Helper function to execute a SELECT query with retry logic."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        return [dict(row) for row in results]
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def find_order_id(text):
    """Extracts a potential order ID from text."""
    match = re.search(r'\b(ORD\d{4,})\b', text, re.IGNORECASE)
    return match.group(1) if match else None

def fetch_order_status(order_id):
    query = "SELECT order_id, customer_name, status FROM order_status WHERE order_id = ?"
    return _execute_query(query, (order_id,))

def fetch_product_info(product_name_keyword):
    query_exact = "SELECT name, features, price FROM product_info WHERE name = ?"
    results_exact = _execute_query(query_exact, (product_name_keyword,))
    if results_exact:
        return results_exact
    query_partial = "SELECT name, features, price FROM product_info WHERE name LIKE ?"
    return _execute_query(query_partial, (f'%{product_name_keyword}%',))

def fetch_support_contacts(department_keyword=None):
    if department_keyword:
        query = "SELECT department, phone, email FROM support_contacts WHERE department LIKE ?"
        return _execute_query(query, (f'%{department_keyword}%',))
    query = "SELECT department, phone, email FROM support_contacts"
    return _execute_query(query)

def fetch_faq(query_text):
    words = query_text.lower().split()
    if not words:
        return []
    conditions = " OR ".join(["question LIKE ? OR keywords LIKE ?"] * len(words))
    query = f"SELECT question, answer FROM faq WHERE {conditions} LIMIT 5"
    params = [f'%{word}%' for word in words for _ in (0, 1)]
    return _execute_query(query, tuple(params))