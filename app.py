import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from PIL import Image
from io import BytesIO
import base64

# Set page config
st.set_page_config(page_title="Gemini Chat & Image Generation", page_icon="🎨", layout="wide")

# App title
st.title("Gemini 2.0 Flash - Chat & Image Generation")

# Initialize session state to store chat history and generated images
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# Load API key from Streamlit secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Key not found in Streamlit secrets. Please add 'GOOGLE_API_KEY' to your secrets.")
    st.stop()

# Configure Google API for chat
genai.configure(api_key=api_key)

# Initialize the model for chat
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# App mode selection
app_mode = st.sidebar.radio("Choose Mode", ["Chat", "Image Generation"])

# Chat Mode
if app_mode == "Chat":
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

# Image Generation Mode
else:
    st.header("Image Generation with Gemini 2.0 Flash")
    
    # Explain why image generation isn't possible
    st.warning(
        "Image generation is not supported by the Google Generative AI library as of May 2025. "
        "The requested model 'gemini-2.0-flash-preview-image-generation' does not exist in the "
        "google.generativeai library, and Google AI does not provide a public API for text-to-image "
        "generation in this context. To proceed with image generation, consider using an alternative provider "
        "like Stability AI (Stable Diffusion) or OpenAI (DALL-E). Would you like to switch to one of these?"
    )

    # Image generation settings
    img_width = st.slider("Image Width", min_value=256, max_value=1024, value=512, step=64)
    img_height = st.slider("Image Height", min_value=256, max_value=1024, value=512, step=64)
    image_prompt = st.text_area("Describe the image you want to generate (e.g., 'bombardiro crocodilo'):", height=100)

    # Generate button
    if st.button("Generate Image"):
        if not image_prompt:
            st.warning("Please enter a description for the image.")
        else:
            st.error(
                "Image generation is not available with Google AI in this library. "
                "The model 'gemini-2.0-flash-preview-image-generation' is not supported."
            )

    # Display generated images (placeholder for when generation is supported)
    if st.session_state.generated_images:
        st.subheader("Generated Images")
        for i, img_data in enumerate(st.session_state.generated_images):
            st.image(
                BytesIO(base64.b64decode(img_data["image"])),
                caption=f"Prompt: {img_data['prompt']} ({img_data['dimensions']})",
                use_column_width=True
            )
            if st.button(f"Remove Image {i+1}", key=f"remove_{i}"):
                st.session_state.generated_images.pop(i)
                st.rerun()

# Instructions in sidebar
with st.sidebar:
    st.markdown("## Instructions")
    
    if app_mode == "Chat":
        st.markdown("1. Type your message in the chat input.")
        st.markdown("2. Type 'quit' to end the session and start a new one.")
        st.markdown("3. Click 'Clear Chat' to reset the conversation.")
    else:
        st.markdown("1. Adjust image size settings.")
        st.markdown("2. Enter a prompt describing the image.")
        st.markdown("3. Note: Image generation is not supported by Google AI in this library.")
    
    st.markdown("---")
    st.markdown("**Note:** API key is loaded from Streamlit secrets.")


