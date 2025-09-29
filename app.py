import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import os
import tempfile
from openai import OpenAI
import gdown
from download_model import download_model


# Load environment variables
from dotenv import load_dotenv
load_dotenv()  


# Load the trained model with error handling
try:
    # 👇 Use download function instead of local file path
    model_path = download_model()
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully from download_model()")
except Exception as e:
    print(f"Failed to download model: {e}")
    try:
        # Try loading from local file
        model = tf.keras.models.load_model("garbage_classification_model_inception.keras")
        print("Model loaded from local file")
    except Exception as e2:
        print(f"Failed to load local model: {e2}")
        # Create a dummy model for demonstration purposes
        print("Creating dummy model for demonstration...")
        model = None

# Define image dimensions
img_height = 384
img_width = 512

# Define dustbin colors
dustbin_colors = {
    'Cardboard': 'brown',
    'Trash': 'grey',
    'Plastic': 'blue',
    'Metal': 'blue',
    'Glass': 'blue',
    'Paper': 'blue'
}

# Initialize OpenAI/OpenRouter client with error handling
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        print("OpenAI client initialized successfully")
    else:
        client = None
        print("OpenAI API key not found, running without AI disposal instructions")
except Exception as e:
    client = None
    print(f"Failed to initialize OpenAI client: {e}")


def predict_waste_category_from_image(image):
    if model is None:
        # Return a demo prediction when model is not available
        return "Plastic (Demo Mode)"
    
    img = image.resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    waste_categories = ['Cardboard', 'Trash', 'Plastic', 'Metal', 'Glass', 'Paper']
    predicted_category_index = np.argmax(prediction)
    predicted_category = waste_categories[predicted_category_index]
    return predicted_category

# Function to get disposal instructions dynamically via API
def get_disposal_instruction(predicted_category):
    if client is None:
        # Return static disposal instructions when API is not available
        disposal_instructions = {
            'Cardboard': 'Cardboard can be recycled! Remove any tape or staples and place in recycling bin.',
            'Trash': 'This item belongs in the general waste bin. Cannot be recycled.',
            'Plastic': 'Most plastic items can be recycled. Check for recycling symbols and clean before recycling.',
            'Metal': 'Metal items are recyclable! Clean them and place in recycling bin.',
            'Glass': 'Glass is recyclable! Rinse clean and place in appropriate recycling container.',
            'Paper': 'Paper can be recycled! Keep it clean and dry for best results.'
        }
        return disposal_instructions.get(predicted_category, 'Please dispose of this item responsibly.')
    
    prompt = f"""
    You are an expert in waste management. Provide **concise, beginner-friendly disposal instructions** for the following waste category: {predicted_category}.
    """
    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3.1:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# Function to get the path of the dustbin image based on color
def get_dustbin_image_path(color):
    dustbin_images_dir = "dustbin_images"
    return os.path.join(dustbin_images_dir, f"{color}.png")

# Function to predict waste category from the 10th frame of a video
def predict_waste_category_from_video(video_bytes):
    temp_file = None
    try:
        # Create a temporary file with proper cleanup
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name

        # Open the video file
        video_cap = cv2.VideoCapture(temp_file_path)

        # Check if the video opened successfully
        if not video_cap.isOpened():
            raise RuntimeError("Error: Failed to open the video file. Please ensure the video is in MP4 format.")

        # Get video properties for better error handling
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 20:
            # For short videos, use the middle frame
            target_frame = total_frames // 2
        else:
            target_frame = 20

        # Skip to target frame
        for i in range(target_frame):
            ret, _ = video_cap.read()
            if not ret:
                # If we can't reach the target frame, use the last available frame
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
                break

        ret, frame = video_cap.read()

        # Release video capture
        video_cap.release()

        # Check if the frame was read successfully
        if ret:
            # Convert the frame to PIL Image format
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            predicted_category = predict_waste_category_from_image(pil_image)

            return predicted_category, frame
        else:
            raise RuntimeError("Error: Failed to read frame from the video.")
            
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the video: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass  # Ignore cleanup errors

# Function to get the path of the dustbin image based on color
def get_dustbin_image_path(color):
    dustbin_images_dir = "dustbin_images"
    return os.path.join(dustbin_images_dir, f"{color}.png")

# Streamlit app
def main():
    st.title("Waste Management AI")

    st.markdown("Upload an image or a video and let the AI classify the waste category.")
    
    # Show demo mode notice if model is not available
    if model is None:
        st.warning("⚠️ Running in Demo Mode - AI model not available. File upload functionality is fully operational!")
    
    # Add file size information for users
    st.info("📁 Supported formats: JPG, JPEG, PNG, MP4 | Maximum file size: 50 MB")

    # File uploader for image or video
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        # Check file size (50MB limit)
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_size_mb > 50:
            st.error(f"❌ File size ({file_size_mb:.1f} MB) exceeds the 50 MB limit. Please upload a smaller file.")
            return
        
        # Show file information
        st.success(f"✅ File uploaded successfully! Size: {file_size_mb:.1f} MB")
        
        # Check if the uploaded file is an image or a video
        if uploaded_file.type.startswith('image'):
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Add processing indicator
            with st.spinner("🔍 Analyzing image... Please wait."):
                # Predict the waste category from the image
                predicted_category = predict_waste_category_from_image(image)
        elif uploaded_file.type.startswith('video'):
            # Show video processing notice
            st.info("🎥 Processing video file. This may take a moment for large files...")
            
            # Display the video
            st.video(uploaded_file, start_time=0)
            
            # Predict the waste category from the first frame of the video
            try:
                with st.spinner("🎬 Extracting frame and analyzing... Please wait."):
                    predicted_category, frame = predict_waste_category_from_video(uploaded_file.read())

            except RuntimeError as e:
                st.error(str(e))
                return
        else:
            st.error("Unsupported file format. Please upload an image (jpg, jpeg, png) or a video (mp4).")
            return

        # Display the prediction
        st.subheader(f"Waste Category Predicted: {predicted_category}")

        # # Get dynamic disposal instructions via API
        st.subheader("Disposal Information:")
        try:
            disposal_text = get_disposal_instruction(predicted_category)
            st.write(disposal_text)
        except Exception as e:
            st.error(f"Error fetching disposal instructions: {str(e)}")

        # Show dustbin image
        dustbin_color = dustbin_colors.get(predicted_category, 'blue')
        dustbin_image_path = get_dustbin_image_path(dustbin_color)
        if os.path.exists(dustbin_image_path):
            dustbin_image = Image.open(dustbin_image_path)
            st.image(dustbin_image, caption=f"Dustbin for {predicted_category}", width=200)

if __name__ == "__main__":

    main()
