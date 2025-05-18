import streamlit as st
import google.generativeai as genai
from io import BytesIO
from PIL import Image # To potentially inspect or save image if needed, not strictly for display

# Set page config
st.set_page_config(page_title="Gemini Image Generation", page_icon="🖼️", layout="wide")

# App title
st.title("Gemini 2.0 - Image Generation")

# Load API key from Streamlit secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Key not found in Streamlit secrets. Please add 'GOOGLE_API_KEY' to your secrets.")
    st.stop()

# Configure API
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API Key is not configured.")
    st.stop()

# --- Image Generation Mode ---
st.header("Generate Images with Gemini")

# Text input for the image generation prompt
prompt = st.text_area("Enter a prompt to generate an image:", height=100, placeholder="e.g., A futuristic cityscape at sunset with flying cars")

if st.button("Generate Image"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image... Please wait."):
            try:
                # Select the model capable of image generation.
                # 'gemini-2.0-flash-preview-image-generation' is one such model.
                # Always check the Google AI documentation for the latest recommended models.
                model = genai.GenerativeModel('gemini-2.0-flash-preview-image-generation')

                # Generate content with response_modalities specified
                # This model can return both text and image, so we request both.
                # Image-only output might not be supported directly; the model often provides context.
                response = model.generate_content(
                    contents=[prompt], # The prompt
                    generation_config=genai.types.GenerationConfig(
                        response_modalities=['IMAGE', 'TEXT'] # Requesting image and text output
                    )
                )

                # Process the response to extract image and text
                generated_image_bytes = None
                generated_text_parts = []

                if response.parts:
                    for part in response.parts:
                        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                            generated_image_bytes = part.inline_data.data
                        elif part.text:
                            generated_text_parts.append(part.text)
                else:
                    # Fallback if response.parts is empty but text might be directly available
                    if response.text:
                         generated_text_parts.append(response.text)


                st.subheader("Generation Result:")
                if generated_image_bytes:
                    st.image(generated_image_bytes, caption="Generated Image")
                else:
                    st.warning("No image was generated. The model might have responded with text only.")

                if generated_text_parts:
                    st.write("**Accompanying Text from Model:**")
                    st.write("\n".join(generated_text_parts))
                elif not generated_image_bytes: # If no image and no text parts, show full response text if any
                     if hasattr(response, 'text') and response.text:
                        st.write(response.text)
                     else:
                        st.error("The model did not return any recognizable content (image or text).")


            except Exception as e:
                st.error(f"An error occurred during image generation: {str(e)}")
                # You might want to print the full response for debugging if an error occurs
                # st.error(f"Full response: {response}")


# Instructions in sidebar
with st.sidebar:
    st.markdown("## Instructions")
    st.markdown("1. Enter a descriptive prompt for the image you want to generate.")
    st.markdown("2. Click 'Generate Image'.")
    st.markdown("3. The generated image and any accompanying text will appear.")
    st.markdown("---")
    st.markdown("**Note on API Usage & Costs:**")
    st.markdown(
        "Image generation uses the Gemini API. While a free tier may be available, "
        "extensive use can incur costs. Please check Google's official Gemini API pricing "
        "for details on free quotas and charges."
    )
    st.markdown("API key is loaded from Streamlit secrets.")
