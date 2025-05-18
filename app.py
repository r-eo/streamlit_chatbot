import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import base64
from PIL import Image
from io import BytesIO

# Set page config
st.set_page_config(page_title="Gemini Chat & Image Analysis", page_icon="🎨", layout="wide")

# App title
st.title("Gemini 2.0 - Chat & Image Analysis")

# Initialize session state to store chat history and analyzed images
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

# Configure API
if api_key:
    genai.configure(api_key=api_key)

# App mode selection
app_mode = st.sidebar.radio("Choose Mode", ["Chat", "Image Analysis"])

# Chat model selection (only visible in chat mode)
if app_mode == "Chat":
    model = st.sidebar.selectbox(
        "Select Gemini Model",
        ["gemini-2.0-flash", "gemini-2.0-pro"],
        index=0
    )

# Clear buttons
if app_mode == "Chat":
    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.messages = []
        st.rerun()

# Main content area based on mode
if app_mode == "Chat":
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    # Process user input for chat
    if user_input:
        # Add user message to displayed messages
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add to langchain chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        # Check if API key is configured correctly
        try:
            # Create progress/thinking indicator
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Initialize the model and get response
                    llm = ChatGoogleGenerativeAI(
                        model=model,
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
else:
    # Image Analysis Mode
    st.header("Image Analysis with Gemini 2.0")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image to analyze:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Prompt input for image analysis
        analysis_prompt = st.text_area("What would you like to know about this image?", height=100, value="Describe this image in detail.")
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                try:
                    # Convert image to base64 for the API
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Use Gemini model for vision (gemini-pro-vision or similar)
                    model = genai.GenerativeModel('gemini-2.0-pro-vision')
                    response = model.generate_content([
                        {"mime_type": "image/png", "data": img_base64},
                        {"text": analysis_prompt}
                    ])
                    
                    st.write("**Analysis Result:**")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Image analysis error: {str(e)}")

# Instructions in sidebar
with st.sidebar:
    st.markdown("## Instructions")
    
    if app_mode == "Chat":
        st.markdown("1. Select the Gemini model you want to use")
        st.markdown("2. Type your message in the chat input")
        st.markdown("3. Click 'Clear Chat' to start a new conversation")
    else:
        st.markdown("1. Upload an image to analyze")
        st.markdown("2. Enter a prompt or question about the image")
        st.markdown("3. Click 'Analyze Image' to get the result")
    
    st.markdown("---")
    st.markdown("**Note:** API key is loaded from Streamlit secrets.")
