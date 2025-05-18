import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import base64
from PIL import Image
from io import BytesIO

# Set page config
st.set_page_config(page_title="Gemini Chat & Image Generation", page_icon="🎨", layout="wide")

# App title
st.title("Gemini 2.0 - Chat & Image Generation")

# Initialize session state to store chat history and generated images
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# API key input
api_key = st.sidebar.text_input("Enter Google API Key", type="password")

# Configure API
if api_key:
    genai.configure(api_key=api_key)

# App mode selection
app_mode = st.sidebar.radio("Choose Mode", ["Chat", "Image Generation"])

# Chat model selection (only visible in chat mode)
if app_mode == "Chat":
    model = st.sidebar.selectbox(
        "Select Gemini Model",
        ["gemini-2.0-flash", "gemini-2.0-pro"],
        index=0
    )

# Image generation model and settings (only visible in image generation mode)
if app_mode == "Image Generation":
    img_model = "gemini-2.0-pro-vision"
    # Image generation settings
    img_width = st.sidebar.slider("Image Width", min_value=256, max_value=1024, value=512, step=64)
    img_height = st.sidebar.slider("Image Height", min_value=256, max_value=1024, value=512, step=64)
    num_images = st.sidebar.slider("Number of Images", min_value=1, max_value=4, value=1)

# Clear buttons
if app_mode == "Chat":
    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.messages = []
        st.rerun()
else:
    if st.sidebar.button("Clear Generated Images"):
        st.session_state.generated_images = []
        st.rerun()

# Function to generate images
def generate_images(prompt, width, height, samples=1):
    try:
        generation_config = {
            "width": width,
            "height": height,
            "samples": samples,
        }
        
        # Use direct genai API for image generation
        response = genai.generate_images(
            model="gemini-2.0-pro-vision",
            prompt=prompt,
            generation_config=generation_config
        )
        
        # Extract the image data
        images = []
        for img_data in response.images:
            # The response format might vary based on the API version
            # Let's handle both base64 and URI formats
            if img_data.startswith("data:image"):
                # Extract base64 part
                img_base64 = img_data.split(",")[1]
                image_bytes = base64.b64decode(img_base64)
            else:
                # Assuming it's already base64
                image_bytes = base64.b64decode(img_data)
                
            image = Image.open(BytesIO(image_bytes))
            images.append(image)
            
        return images
    except Exception as e:
        st.error(f"Image generation error: {str(e)}")
        return []

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
else:
    # Image Generation Mode
    st.header("Image Generation with Gemini 2.0")
    
    # Prompt input for image generation
    image_prompt = st.text_area("Describe the image you want to generate:", height=100)
    
    # Generate button
    if st.button("Generate Image(s)"):
        if not api_key:
            st.error("Please enter your Google API key in the sidebar.")
        elif not image_prompt:
            st.warning("Please enter a description for the image.")
        else:
            with st.spinner(f"Generating {num_images} image(s)..."):
                images = generate_images(image_prompt, img_width, img_height, num_images)
                if images:
                    # Store the generated images
                    new_images = []
                    for img in images:
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        new_images.append({
                            "image": img_str,
                            "prompt": image_prompt,
                            "dimensions": f"{img_width}x{img_height}"
                        })
                    
                    st.session_state.generated_images = new_images + st.session_state.generated_images
    
    # Display generated images
    if st.session_state.generated_images:
        st.subheader("Generated Images")
        
        # Create rows with 2 images per row
        for i in range(0, len(st.session_state.generated_images), 2):
            cols = st.columns(2)
            for j in range(2):
                if i+j < len(st.session_state.generated_images):
                    img_data = st.session_state.generated_images[i+j]
                    with cols[j]:
                        st.image(
                            BytesIO(base64.b64decode(img_data["image"])), 
                            caption=f"Prompt: {img_data['prompt']} ({img_data['dimensions']})",
                            use_column_width=True
                        )
                        if st.button(f"Remove", key=f"remove_{i+j}"):
                            st.session_state.generated_images.pop(i+j)
                            st.rerun()

# Instructions in sidebar
with st.sidebar:
    st.markdown("## Instructions")
    st.markdown("1. Enter your Google API key in the sidebar")
    
    if app_mode == "Chat":
        st.markdown("2. Select the Gemini model you want to use")
        st.markdown("3. Type your message in the chat input")
        st.markdown("4. Click 'Clear Chat' to start a new conversation")
    else:
        st.markdown("2. Adjust image size and number settings")
        st.markdown("3. Enter a detailed prompt describing the image")
        st.markdown("4. Click 'Generate Image(s)' to create images")
        st.markdown("5. Images are stored in your session until cleared")
    
    st.markdown("---")
    st.markdown("**Note:** Image generation requires access to the Gemini 2.0 models with image generation capabilities.")
