import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

st.set_page_config(page_title="CS Teacher Chatbot", page_icon="💻")
st.title("💻 CS Teacher Chatbot")

SYSTEM_PROMPT = "You are an experienced Computer Science Teacher"


@st.cache_resource
def get_model():
    return init_chat_model("mistralai:mistral-small-2603")


model = get_model()

# Initialize chat history in session state (persists across reruns)
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# Render existing chat history (skip the system message)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input box at the bottom
user_input = st.chat_input("Ask me anything about Computer Science...")

if user_input:
    # Show and store user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
        st.markdown(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))

# Sidebar controls
with st.sidebar:
    st.header("Options")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        st.rerun()

    if st.button("📄 Show raw message history"):
        st.write(st.session_state.messages)