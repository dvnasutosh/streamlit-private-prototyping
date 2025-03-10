import streamlit as st
import tempfile
import os

# ---------------------- PDF RETRIEVAL IMPORTS ---------------------- #
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------- CHAT MODEL IMPORT ---------------------- #
from langchain_groq import ChatGroq

# ---------------------- PDF RETRIEVAL FUNCTIONS ---------------------- #
def save_uploaded_pdf(uploaded_file):
    """Save the uploaded PDF to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def load_pdf_documents(pdf_path):
    """Load documents from a PDF using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunks for improved retrieval."""
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_embeddings(model_name="all-MiniLM-L6-v2"):
    """Create a SentenceTransformerEmbeddings object with the specified model."""
    return SentenceTransformerEmbeddings(model_name=model_name)

def build_retriever(docs, embeddings):
    """Build an in-memory vector retriever from document chunks using FAISS."""
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()

def perform_search(retriever, query):
    """Retrieve relevant documents based on the search query."""
    return retriever.get_relevant_documents(query)

def pdf_retrieval():
    st.header("PDF Retrieval")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        pdf_path = save_uploaded_pdf(uploaded_file)
        st.info("Processing PDF...")
        
        documents = load_pdf_documents(pdf_path)
        st.write(f"Loaded {len(documents)} document(s) from the PDF.")
        
        docs = split_documents(documents)
        st.write(f"Split document into {len(docs)} chunks.")
        
        embeddings = create_embeddings()
        retriever = build_retriever(docs, embeddings)
        st.success("PDF processed and vectorized in-memory. You can now search the content.")
        
        query = st.text_input("Enter your search query:")
        if st.button("Search"):
            if query:
                results = perform_search(retriever, query)
                st.write("### Search Results")
                for idx, doc in enumerate(results, start=1):
                    st.markdown(f"**Result {idx}:**")
                    st.write(doc.page_content)
            else:
                st.error("Please enter a query.")
        
        # Clean up the temporary file
        os.remove(pdf_path)

# ---------------------- CHAT INTERFACE FUNCTIONS ---------------------- #
def load_chat_model():
    """Instantiate the ChatGroq model using the API key from secrets."""
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in secrets.toml!")
        return None
    # Set the environment variable so that ChatGroq picks it up.
    os.environ["GROQ_API_KEY"] = api_key
    # Instantiate ChatGroq with desired parameters.
    return ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

def chat_with_model(model, new_message, history):
    """
    Update the conversation history with the new user message and the model's reply.
    The conversation history is a list of tuples: (role, message).
    """
    # Append the human message.
    history.append(("human", new_message))
    # Invoke the model with the current conversation history.
    # The ChatGroq model expects a list of (role, message) tuples.
    ai_msg = model.invoke(history)
    # Append the assistant reply.
    history.append(("assistant", ai_msg.content))
    return history

def chat_interface():
    st.header("Groq Chat")
    model = load_chat_model()
    if model is None:
        return

    # Initialize conversation history in session state.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("Conversation")
    for role, message in st.session_state.chat_history:
        if role == "human":
            st.markdown(f"**User:** {message}")
        else:
            st.markdown(f"**Assistant:** {message}")

    new_message = st.text_input("Enter your message:", key="new_message")
    if st.button("Send"):
        if new_message:
            st.session_state.chat_history = chat_with_model(model, new_message, st.session_state.chat_history)
            st.rerun()
        else:
            st.error("Please enter a message.")

# ---------------------- MAIN APP FUNCTION ---------------------- #
def main():
    st.title("Multi-Function App: PDF Retrieval & Chat")
    
    # Sidebar mode selector.
    mode = st.sidebar.radio("Select Mode", ["PDF Retrieval", "Chat"])
    
    if mode == "PDF Retrieval":
        pdf_retrieval()
    elif mode == "Chat":
        chat_interface()

if __name__ == "__main__":
    main()
