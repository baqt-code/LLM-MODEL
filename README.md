# Sahara: News Research Tool 📈

Welcome to Sahara, a news research tool that helps you process and analyze news articles using state-of-the-art machine learning models. This tool allows you to input URLs of news articles, process them, and perform question-answering on the processed data.

## Features

- **User Authentication**: Secure login and registration system for users.
- **Data Loading**: Load news articles from provided URLs.
- **Text Splitting**: Split the loaded articles into manageable chunks.
- **Embeddings and Vector Store**: Create embeddings for the chunks and store them in a FAISS index.
- **Question Answering**: Use a pre-trained question-answering model to answer questions based on the processed articles.

## Installation

To run Sahara locally, follow these steps:

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Authentication**:
    - When you first access the tool, you'll be presented with a login form. If you don't have an account, you can register using the registration form.
    - Enter your username and password to log in.

2. **Processing URLs**:
    - Once logged in, enter up to three URLs of news articles in the sidebar.
    - Click the "Process URLs" button to load and process the articles. The tool will load the data, split it into chunks, create embeddings, and save the FAISS index.

3. **Question Answering**:
    - Enter your query in the "Question" input box.
    - If the FAISS index exists and documents are available, the tool will use the pre-trained question-answering model to find the best answer to your question.

## File Structure

- `app.py`: Main application file containing the Streamlit app code.
- `user_db.pkl`: Placeholder for user database (stored securely in a real-world application).
- `faiss_store_huggingface.pkl`: FAISS index file created after processing URLs.
- `requirements.txt`: List of dependencies required to run the application.

