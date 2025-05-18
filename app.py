import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Set page config
st.set_page_config(page_title="Gemini Chat", page_icon="💬")

# App title
st.title("Chat with Gemini")

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

# API key input
api_key = st.sidebar.text_input("Enter Google API Key", type="password")

# Model selection
model = st.sidebar.selectbox(
    "Select Gemini Model",
    ["gemini-2.0-flash", "gemini-2.0-pro"],
    index=0
)

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]
    st.session_state.messages = []
    st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

# Process user input
if user_input:
    # Add user message to displayed messages
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Add to langchain chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Check if API key is provided
    if not api_key:
        st.error("Please enter your Google API key in the sidebar.")
    else:
        try:
            # Create progress/thinking indicator
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Initialize the model and get response
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=api_key
                    )
                    result = llm.invoke(st.session_state.chat_history)
                    
                    # Add to langchain chat history
                    st.session_state.chat_history.append(AIMessage(content=result.content))
                    
                    # Add to displayed messages
                    st.session_state.messages.append({"role": "assistant", "content": result.content})
                    
                    # Display the response
                    st.write(result.content)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Instructions in sidebar
with st.sidebar:
    st.markdown("## Instructions")
    st.markdown("1. Enter your Google API key in the sidebar")
    st.markdown("2. Select the Gemini model you want to use")
    st.markdown("3. Type your message in the chat input")
    st.markdown("4. Click 'Clear Chat' to start a new conversation")
