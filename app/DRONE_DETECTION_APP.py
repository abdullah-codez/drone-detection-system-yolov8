import streamlit as st
import tempfile
import os
import glob
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Drone Detection", page_icon="🛸", layout="wide")

st.title("🛸 Drone Detection in Images and Footages")
st.markdown("Upload a video or image and detect drones using YOLOv8. Download the processed output after detection.")

@st.cache_resource
def load_model():
    model_path = "Yolov8m_trained_wights_NEW.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    return YOLO(model_path)

model = load_model()

# File uploader for both videos and images
uploaded_file = st.file_uploader(
    "🎥 Upload a Drone Video or Image", 
    type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "bmp"]
)

def process_video(input_video_path, uploaded_file_name):
    """Process video file and return output path"""
    st.info("🔍 Running drone detection on video... please wait.")

    # Run YOLO prediction
    results = model.predict(
        source=input_video_path,
        conf=0.3,
        save=False  # We'll handle saving manually
    )

    # Open original video to extract frame size and FPS
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output path
    output_dir = os.path.join("runs", "detect")
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"processed_{uploaded_file_name}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write detected frames manually
    for r in results:
        frame = r.plot()  # Draw boxes, labels, etc.
        out.write(frame)

    cap.release()
    out.release()
    
    return output_video_path

def process_image(input_image_path, uploaded_file_name):
    """Process image file and return output path"""
    st.info("🔍 Running drone detection on image... please wait.")

    # Run YOLO prediction
    results = model.predict(
        source=input_image_path,
        conf=0.3,
        save=False
    )

    # Prepare output path
    output_dir = os.path.join("runs", "detect")
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, f"processed_{uploaded_file_name}")

    # Process and save the image
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save(output_image_path)  # save image
    
    return output_image_path

if uploaded_file:
    # Determine file type
    file_extension = uploaded_file.name.lower().split('.')[-1]
    is_video = file_extension in ['mp4', 'avi', 'mov', 'mkv']
    is_image = file_extension in ['jpg', 'jpeg', 'png', 'bmp']
    
    if not (is_video or is_image):
        st.error("❌ Unsupported file format. Please upload a video or image file.")
        st.stop()

    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
    tfile.write(uploaded_file.read())
    input_file_path = tfile.name
    tfile.close()

    try:
        if is_video:
            # Process video
            output_path = process_video(input_file_path, uploaded_file.name)
            
            if os.path.exists(output_path):
                st.success("✅ Video detection complete! Your processed video is ready to download.")
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Output Video",
                        data=f,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_output.mp4",
                        mime="video/mp4"
                    )
                
                # Display a preview of the processed video
                st.video(output_path)
                
            else:
                st.error("❌ Could not generate output video.")

        else:  # Image processing
            # Process image
            output_path = process_image(input_file_path, uploaded_file.name)
            
            if os.path.exists(output_path):
                st.success("✅ Image detection complete! Your processed image is ready to download.")
                
                # Display the processed image
                st.image(output_path, caption="Processed Image with Detections", use_container_width=True)
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Output Image",
                        data=f,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_output.jpg",
                        mime="image/jpeg"
                    )
            else:
                st.error("❌ Could not generate output image.")

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")

    finally:
        # Clean up temporary input file
        try:
            os.remove(input_file_path)
        except PermissionError:
            pass

else:
    st.info("⬆️ Please upload a video or image to start detection.")