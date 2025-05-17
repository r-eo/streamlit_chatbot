import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import os

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

# Initialize LangChain LLM for chat
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize generative AI client for image generation
genai_client = genai.GenerativeAI()

# Streamlit app title
st.title("Gemini Chat and Image Generation App")

# Chat interface
st.subheader("Chat with Gemini")
user_input = st.text_input("Your message:", key="chat_input", placeholder="Type your message or 'quit' to clear chat...")

if st.button("Send Message"):
    if user_input:
        if user_input.lower() == "quit":
            st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]
            st.success("Chat history cleared!")
        else:
            # Add user message to chat history
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            
            try:
                # Get AI response
                result = llm.invoke(st.session_state.chat_history)
                st.session_state.chat_history.append(AIMessage(content=result.content))
            except Exception as e:
                st.error(f"Error in chat: {str(e)}")

# Display chat history
st.write("### Chat History")
for message in st.session_state.chat_history:
    if isinstance(message, SystemMessage):
        continue
    role = "You" if isinstance(message, HumanMessage) else "AI"
    with st.chat_message(role.lower()):
        st.write(message.content)

# Image generation interface
st.subheader("Generate an Image")
image_prompt = st.text_input("Enter prompt for image generation:", key="image_input", placeholder="e.g., 'bombardiro crocodilo'")
if st.button("Generate Image"):
    if image_prompt:
        try:
            # Generate image
            response = genai_client.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=image_prompt,
                generation_config={
                    "response_modalities": ["TEXT", "IMAGE"]
                }
            )
            
            # Process response
            for part in response.candidates[0].content.parts:
                if part.text:
                    st.write("**Generated Text:**")
                    st.write(part.text)
                if part.inline_data:
                    # Decode and display image
                    image_data = part.inline_data.data
                    image = Image.open(BytesIO(image_data))
                    st.image(image, caption="Generated Image", use_column_width=True)
                    
                    # Save image for download
                    image.save("generated_image.png")
                    with open("generated_image.png", "rb") as file:
                        st.download_button(
                            label="Download Image",
                            data=file,
                            file_name="generated_image.png",
                            mime="image/png",
                            key="download_button"
                        )
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")

# Instructions
st.markdown("""
### Instructions
- **Chat**: Enter a message in the chat input and click "Send Message". Type 'quit' to clear the chat history.
- **Image Generation**: Enter a prompt in the image input and click "Generate Image" to create an image.
- **Download**: Use the download button to save generated images.
- **Note**: Ensure `GOOGLE_API_KEY` is set in your environment variables or Streamlit secrets.
""")
