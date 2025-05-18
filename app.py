import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from PIL import Image
from io import BytesIO
import base64
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

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

# Load API keys from Streamlit secrets
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    stability_api_key = st.secrets["STABILITY_API_KEY"]
except KeyError as e:
    st.error(f"API Key not found in Streamlit secrets: {str(e)}. Please add 'GOOGLE_API_KEY' and 'STABILITY_API_KEY' to your secrets.")
    st.stop()

# Configure Google API for chat
genai.configure(api_key=google_api_key)

# Initialize the model for chat
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

# Configure Stability AI API for image generation
stability_api = client.StabilityInference(
    key=stability_api_key,
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0",  # Use the latest Stable Diffusion model
)

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
    st.header("Image Generation with Stable Diffusion")
    
    # Image generation settings
    img_width = st.slider("Image Width", min_value=256, max_value=1024, value=512, step=64)
    img_height = st.slider("Image Height", min_value=256, max_value=1024, value=512, step=64)
    num_images = st.slider("Number of Images", min_value=1, max_value=4, value=1)
    image_prompt = st.text_area("Describe the image you want to generate (e.g., 'bombardiro crocodilo'):", height=100)

    # Function to generate images using Stability AI
    def generate_images(prompt, width, height, samples=1):
        try:
            images = []
            for _ in range(samples):
                # Generate image using Stability AI
                response = stability_api.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    steps=50,  # Number of diffusion steps (higher for better quality)
                    cfg_scale=7.0,  # How closely the image follows the prompt
                    samples=1,  # Number of images per generation
                )
                
                # Process the response
                for resp in response:
                    for artifact in resp.artifacts:
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            image = Image.open(BytesIO(artifact.binary))
                            images.append(image)
                        else:
                            st.warning("Non-image artifact received.")
                            
            return images
        except Exception as e:
            st.error(f"Image generation error: {str(e)}")
            return []

    # Generate button
    if st.button("Generate Image(s)"):
        if not image_prompt:
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
    
    if app_mode == "Chat":
        st.markdown("1. Type your message in the chat input.")
        st.markdown("2. Type 'quit' to end the session and start a new one.")
        st.markdown("3. Click 'Clear Chat' to reset the conversation.")
    else:
        st.markdown("1. Adjust image size and number settings.")
        st.markdown("2. Enter a prompt describing the image.")
        st.markdown("3. Click 'Generate Image(s)' to create images.")
        st.markdown("4. Images are stored in your session until cleared.")
    
    st.markdown("---")
    st.markdown("**Note:** API keys are loaded from Streamlit secrets.")

