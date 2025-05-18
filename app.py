import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from google.ai.generativelanguage_v1beta.types import GenerateContentResponse
from google.ai import generativelanguage as genai
from PIL import Image
from io import BytesIO
import base64

# Initialize Streamlit page
st.set_page_config(page_title="Gemini Chat + Image Gen", layout="wide")
st.title("💬 Gemini Chat + 🖼️ Image Generation")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]

# Load API key from secrets or input
if "GOOGLE_API_KEY" not in st.secrets:
    st.sidebar.warning("API Key not found in secrets.toml")
    google_api_key = st.sidebar.text_input("Enter Google API Key:", type="password")
else:
    google_api_key = st.secrets["GOOGLE_API_KEY"]

if not google_api_key:
    st.warning("Please provide an API key to proceed.")
    st.stop()

# LangChain LLM for regular chat
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
parser = StrOutputParser()

# Native Gemini Client for image generation
client = genai.Client(api_key=google_api_key)

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("human"):
        st.markdown(user_input)

    # Check if it's an image generation request
    if any(keyword in user_input.lower() for keyword in ["generate image", "create image", "draw image"]):
        with st.spinner("Generating image..."):
            try:
                # Call Gemini 2.0 Flash (image generation model)
                response = client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=user_input,
                    config=genai.types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"]
                    )
                )

                # Process parts of response
                ai_response = ""
                image_found = False
                for part in response.candidates[0].content.parts:
                    if part.text:
                        ai_response += part.text + "\n\n"
                    elif part.inline_data:
                        image_bytes = part.inline_data.data
                        image = Image.open(BytesIO(image_bytes))
                        st.image(image, caption="Generated Image", use_column_width=True)
                        image.save("generated_image.png")
                        image_found = True

                if ai_response.strip():
                    st.markdown(f"**Assistant:**\n{ai_response}")
                if not ai_response and not image_found:
                    st.markdown("No response received.")

                # Save AI message to chat history
                st.session_state.chat_history.append(AIMessage(content=ai_response))

            except Exception as e:
                st.error(f"Error generating image: {e}")

    else:
        # Regular chat response
        with st.spinner("Thinking..."):
            try:
                result = llm.invoke(st.session_state.chat_history)
                ai_response = parser.invoke(result)
                st.session_state.chat_history.append(AIMessage(content=ai_response))

                with st.chat_message("assistant"):
                    st.markdown(ai_response)
            except Exception as e:
                st.error(f"Error invoking model: {e}")
