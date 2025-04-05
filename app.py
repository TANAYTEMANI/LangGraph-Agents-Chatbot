import uuid
import logging
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from main import process_user_query


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.set_page_config(page_title="PDF Chatbot/Weather.", layout="centered")
st.title("📄 Chat with your PDF or Know your Weather.")
st.markdown("Upload a PDF and ask questions based on its content.")

# Session state for chat and vector store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# PDF upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# Process PDF and store vector_store once
if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        try:
            # Load and split
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                encode_kwargs={"normalize_embeddings": True},
            )
            vector_store = FAISS.from_documents(chunks, embedding=embeddings)

            st.success("PDF processed and ready to chat!")

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

# User question
user_query = st.text_input("Ask a question about your PDF or Weather.")

if st.button("Submit"):
    with st.spinner("Generating answer..."):
        try:
            # Use existing vector store
            answer = process_user_query(
                user_query, st.session_state.thread_id, vector_store
            )
            final_answer = answer["messages"][-1].content

            st.session_state.chat_history.append(("🧑", user_query))
            st.session_state.chat_history.append(("🤖", final_answer))
            st.markdown(final_answer)

        except Exception as e:
            st.error(f"Error: {str(e)}")
