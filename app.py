import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import google.generativeai as genai
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

# Configure Google API key from Streamlit secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])



# Initialize session state for chat history and image display
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

# Streamlit app title
st.title("Chat with AI Assistant (Text & Image Generation)")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Display previously generated images
if st.session_state.generated_images:
    st.subheader("Generated Images")
    for img_path, prompt in st.session_state.generated_images:
        st.image(img_path, caption=f"Generated for: {prompt}", use_column_width=True)

# Input box for user message
user_input = st.chat_input("Type your message here (e.g., 'bombardiro crocodilo')...")

if user_input:
    if user_input.lower() == "quit":
        # Reset chat history and images
        st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]
        st.session_state.generated_images = []
        st.experimental_rerun()
    else:
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response from Google Generative AI
        try:
            response = client.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=user_input,
                generation_config={
                    "response_mime_types": ["text/plain", "image/png"]
                }
            )
            
            # Process response
            text_response = ""
            image_path = None
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_response += part.text
                elif hasattr(part, 'inline_data') and part.inline_data:
                    # Decode and save the image
                    image_data = base64.b64decode(part.inline_data.data)
                    image = Image.open(BytesIO(image_data))
                    image_path = f"generated_image_{uuid.uuid4()}.png"
                    image.save(image_path)
                    st.session_state.generated_images.append((image_path, user_input))
            
            # Display text response
            if text_response:
                with st.chat_message("assistant"):
                    st.write(text_response)
                st.session_state.chat_history.append(AIMessage(content=text_response))
            
            # Refresh to show new image
            if image_path:
                st.experimental_rerun()
                
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error generating response: {str(e)}")
            st.session_state.chat_history.append(AIMessage(content=f"Error: {str(e)}"))

# Clean up old images (optional, to manage disk space)
if len(st.session_state.generated_images) > 10:
    old_image_path, _ = st.session_state.generated_images.pop(0)
    if os.path.exists(old_image_path):
        os.remove(old_image_path)
