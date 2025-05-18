import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Set page config
st.set_page_config(page_title="Gemini Chat", page_icon="💬", layout="wide")

# App title
st.title("Gemini 2.0 Flash - Chat Application")

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load API key from Streamlit secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Key not found in Streamlit secrets. Please add 'GOOGLE_API_KEY' to your secrets.")
    st.stop()

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

# Process user input for chat
if user_input:
    if user_input.lower() == "quit":
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.messages = []
        st.success("Chat session ended. Start a new conversation!")
        st.rerun()

    # Add user message to displayed messages
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Add to langchain chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get response from the model
    try:
        with st.spinner("Thinking..."):
            result = llm.invoke(st.session_state.chat_history)

            # Add to langchain chat history
            st.session_state.chat_history.append(AIMessage(content=result.content))

            # Add to displayed messages
            st.session_state.messages.append({"role": "assistant", "content": result.content})

            # Display the response
            with st.chat_message("assistant"):
                st.write(result.content)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]
    st.session_state.messages = []
    st.rerun()

# Instructions
st.sidebar.markdown("## Instructions")
st.sidebar.markdown("1. Type your message in the chat input.")
st.sidebar.markdown("2. Type 'quit' to end the session and start a new one.")
st.sidebar.markdown("3. Click 'Clear Chat' to reset the conversation.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** API key is loaded from Streamlit secrets.")
