import os
import streamlit as st
import pickle
import time
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Placeholder for user database (in real-world use a secure database)
user_db = {"admin": "password"}

def save_user_db():
    with open("user_db.pkl", "wb") as f:
        pickle.dump(user_db, f)

def load_user_db():
    global user_db
    if os.path.exists("user_db.pkl"):
        with open("user_db.pkl", "rb") as f:
            user_db = pickle.load(f)

# Load user database
load_user_db()

# Function to display the login form
def show_login_form():
    st.write(
        """
        <style>
        .login-form {
            background: rgba(255, 255, 255, 0.7);
            padding: 20px;
            border-radius: 10px;
            max-width: 400px;
            margin: auto;
        }
        .login-form input {
            color: #000;
        }
        .login-form button {
            color: #000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    login_form = st.form("login_form", clear_on_submit=True)
    login_form.markdown("<h1 style='text-align: center; color: black;'>Login</h1>", unsafe_allow_html=True)
    username = login_form.text_input("Username", value="", key="login_username", help="Enter your username")
    password = login_form.text_input("Password", value="", type="password", key="login_password", help="Enter your password")
    login_button = login_form.form_submit_button("Login", help="Click to login")

    if login_button:
        if username in user_db and user_db[username] == password:
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password")

# Function to display the registration form
def show_registration_form():
    st.write(
        """
        <style>
        .register-form {
            background: rgba(255, 255, 255, 0.7);
            padding: 20px;
            border-radius: 10px;
            max-width: 400px;
            margin: auto.
        }
        .register-form input {
            color: #000.
        }
        .register-form button {
            color: #000.
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    register_form = st.form("register_form", clear_on_submit=True)
    register_form.markdown("<h1 style='text-align: center; color: black;'>Register</h1>", unsafe_allow_html=True)
    username = register_form.text_input("Username", value="", key="register_username", help="Enter your username")
    password = register_form.text_input("Password", value="", type="password", key="register_password", help="Enter your password")
    register_button = register_form.form_submit_button("Register", help="Click to register")

    if register_button:
        if username in user_db:
            st.error("Username already exists. Please choose a different username.")
        else:
            user_db[username] = password
            save_user_db()
            st.success("Registration successful. Please login.")

# Add background image and change font color
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/11034868/pexels-photo-11034868.jpeg");
        background-size: cover;
        background-position: center;
        color: black.
    }
    .stTextInput label, .stTextArea label, .stButton button {
        color: black.
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Check if the user is logged in
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Display either login form or registration form
if not st.session_state.logged_in:
    show_login_form()
    st.markdown("---")
    show_registration_form()
else:
    st.title("Sahara: News Research Tool 📈")
    st.sidebar.title("News Article URLs")

    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
        urls.append(url)

    process_url_clicked = st.sidebar.button("Process URLs")
    file_path = "faiss_store_huggingface.pkl"

    main_placeholder = st.empty()

    docs = []  # Initialize docs to avoid NameError

    if process_url_clicked:
        if not any(urls):
            main_placeholder.text("Please enter at least one URL.")
        else:
            try:
                # Load data
                loader = UnstructuredURLLoader(urls=urls)
                main_placeholder.text("Data Loading...Started...✅✅✅")
                data = loader.load()
                st.write(f"Number of documents loaded: {len(data)}")

                # Split data
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                main_placeholder.text("Text Splitter...Started...✅✅✅")
                docs = text_splitter.split_documents(data)
                st.write(f"Number of chunks created: {len(docs)}")

                # Create embeddings and save it to FAISS index
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vectorstore_huggingface = FAISS.from_documents(docs, embeddings)
                main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                time.sleep(2)

                # Save the FAISS index to a pickle file
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_huggingface, f)
                st.write("FAISS index saved successfully.")

            except Exception as e:
                main_placeholder.text(f"Error processing URLs: {e}")

    # Check if query is entered
    query = st.text_input("Question: ", key="query")
    if query and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            qa_pipeline = pipeline("question-answering", model=AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad"), tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad"))
            llm = HuggingFacePipeline(pipeline=qa_pipeline)

            if docs:  # Only process if docs are not empty
                results = []
                for doc in docs:
                    context = doc.page_content
                    inputs = {"question": query, "context": context}
                    result = qa_pipeline(inputs)
                    results.append(result)

                if results:
                    best_answer = max(results, key=lambda x: x['score'])
                    st.header("Answer")
                    st.write(best_answer['answer'])

                    # Display sources, if available
                    sources = best_answer.get("source", "")
                    if sources:
                        st.subheader("Sources:")
                        st.write(sources)
                else:
                    st.write("No answers found.")
            else:
                st.write("No documents available to process.")
    else:
        if query:
            st.write("FAISS index not found. Please process the URLs first.")
