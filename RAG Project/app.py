"""
Streamlit RAG Chatbot
----------------------
Upload a PDF, build a Chroma vector store from it using Mistral embeddings,
and chat with the document using ChatMistralAI (mistral-small-2603).

Run with:
    streamlit run app.py

Requires a .env file (or environment variable) with:
    MISTRAL_API_KEY=your_key_here
"""

import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Chat with your Document", page_icon="📄", layout="wide")
st.title("📄 Chat with your Document")
st.caption("Upload a PDF, then ask questions about it. Powered by Mistral + LangChain + Chroma.")

# --------------------------------------------------------------------------
# Prompt template (same as underlying logic)
# --------------------------------------------------------------------------
TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """
    Context:
    {context}

    Question:
    {question}
    """,
        ),
    ]
)

# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None


@st.cache_resource(show_spinner=False)
def get_llm():
    """Cache the LLM instance across reruns."""
    return ChatMistralAI(model="mistral-small-2603")


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Cache the embeddings instance across reruns."""
    return MistralAIEmbeddings(model="mistral-embed")


def process_document(uploaded_file, chunk_size, chunk_overlap, k, fetch_k):
    """Load, split, embed, and store the uploaded PDF in a fresh Chroma collection."""

    # Clean up any previous persist directory so old chunks don't leak in
    if st.session_state.persist_dir and os.path.exists(st.session_state.persist_dir):
        shutil.rmtree(st.session_state.persist_dir, ignore_errors=True)

    # Save the uploaded file to a temp path so PyPDFLoader can read it
    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    persist_dir = os.path.join(tmp_dir, "chroma_langchain_db")
    st.session_state.persist_dir = persist_dir

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="docs",
        persist_directory=persist_dir,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k}
    )

    st.session_state.vector_store = vector_store
    st.session_state.retriever = retriever
    st.session_state.processed_file = uploaded_file.name
    st.session_state.messages = []  # reset chat when a new doc is processed

    return len(chunks)


def answer_query(query):
    """Retrieve context and get an answer from the LLM."""
    retriever = st.session_state.retriever
    llm = get_llm()

    context_docs = retriever.invoke(query)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    prompt = TEMPLATE.format(context=context_text, question=query)
    response = llm.invoke(prompt)
    return response.content, context_docs


# --------------------------------------------------------------------------
# Sidebar: upload + settings + processing
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    with st.expander("Advanced settings"):
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=10, step=10)
        k = st.slider("Chunks to use per answer (k)", min_value=1, max_value=10, value=4)
        fetch_k = st.slider("Chunks to fetch before MMR (fetch_k)", min_value=k, max_value=30, value=10)

    process_clicked = st.button("Process document", type="primary", disabled=uploaded_file is None)

    if process_clicked and uploaded_file is not None:
        with st.spinner("Reading, splitting, and embedding your document..."):
            try:
                num_chunks = process_document(uploaded_file, chunk_size, chunk_overlap, k, fetch_k)
                st.success(f"Processed '{uploaded_file.name}' into {num_chunks} chunks. Ready to chat!")
            except Exception as e:
                st.error(f"Something went wrong while processing the document: {e}")

    if st.session_state.processed_file:
        st.info(f"Currently loaded: **{st.session_state.processed_file}**")

    if st.session_state.messages:
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()

# --------------------------------------------------------------------------
# Main area: chat interface
# --------------------------------------------------------------------------
if st.session_state.retriever is None:
    st.info("👈 Upload a PDF and click **Process document** to get started.")
else:
    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask a question about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = answer_query(query)
                except Exception as e:
                    answer = f"Sorry, I ran into an error: {e}"
                    sources = []
                st.markdown(answer)

                if sources:
                    with st.expander("View source chunks used"):
                        for i, doc in enumerate(sources, start=1):
                            page = doc.metadata.get("page", "N/A")
                            st.markdown(f"**Chunk {i} (page {page}):**")
                            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

        st.session_state.messages.append({"role": "assistant", "content": answer})