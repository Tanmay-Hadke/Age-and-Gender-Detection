import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import io
import tensorflow as tf
import datetime
import sys
import time
import os

# Set page config at the very beginning
st.set_page_config(
    page_title="Age and Gender Detection",
    layout="wide"
)

# Hide Streamlit branding
st.markdown("""
    <style>
    #MainMenu {
        display: none;
    }
    .stDeployButton {
        display: none;
    }
    header {
        display: none;
    }
    footer {
        display: none;
    }
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 10px;
        text-align: center;
        font-size: 14px;
        color: white;
        background-color: transparent;
    }
    .custom-footer span {
        color: white;
    }
    .title {
        text-align: center;
        color: #2e7d32;
        padding-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #1976d2;
        font-style: italic;
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Set the default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')
import locale
locale.setlocale(locale.LC_ALL, 'C')

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        background-color: #0e1117;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #1f2937;
    }
    .header h1 {
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .header p {
        color: #c2c7d0;
        font-size: 1.1rem;
    }
    .stTabs {
        background: #0e1117;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #1f2937;
    }
    .webcam-controls {
        background: #0e1117;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #1f2937;
        margin: 1rem 0;
    }
    .section {
        background: #0e1117;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #1f2937;
    }
    .stButton button {
        background-color: #00cc66;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #00b359;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    footer {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detection_models():
    """Load the age and gender detection models."""
    # Set mixed precision policy
    tf.keras.mixed_precision.set_global_policy('float32')
    
    # Load models with custom_objects if needed
    model_age = load_model('best_model.keras', compile=False)
    model_gender = load_model('best_model_gender.keras', compile=False)
    
    # Compile models with appropriate settings
    model_age.compile(optimizer='adam', loss='mse')
    model_gender.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model_age, model_gender

def get_age_bin(age):
    """Categorize age into specific bins."""
    if age < 5:
        return '0-5'
    elif 5 <= age < 10:
        return '5-10'
    elif 10 <= age < 15:
        return '10-15'
    elif 15 <= age < 20:
        return '15-20'
    elif 20 <= age < 25:
        return '20-25'
    elif 25 <= age < 30:
        return '25-30'
    elif 30 <= age < 35:
        return '30-35'
    elif 35 <= age < 40:
        return '35-40'
    elif 40 <= age < 45:
        return '40-45'
    elif 45 <= age < 50:
        return '45-50'
    elif 50 <= age < 55:
        return '50-55'
    elif 55 <= age < 60:
        return '55-60'
    elif 60 <= age < 65:
        return '60-65'
    elif 65 <= age < 70:
        return '65-70'
    elif 70 <= age < 75:
        return '70-75'
    elif 75 <= age < 80:
        return '75-80'
    elif 80 <= age < 85:
        return '80-85'
    elif 85 <= age < 90:
        return '85-90'
    elif 90 <= age < 95:
        return '90-95'
    elif 95 <= age < 100:
        return '95-100'
    else:
        return '100+'

def process_image(image_bytes):
    """Process the uploaded image and detect age and gender."""
    # Convert uploaded image to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save temporary file for face detection
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, img)
    
    # Load models
    model_age, model_gender = load_detection_models()
    label_gender = ['Male', 'Female']
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Debugging: Print the output of face_detection
    st.write(f"Detected boxes: {faces}")

    # Ensure box is not empty before processing
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Prepare face region for prediction
            face_img = cv2.resize(img[y:y+h, x:x+w], dsize=(50, 50))
            face_img = face_img.reshape(1, 50, 50, 3) / 255.
            
            try:
                # Predict age with error handling
                age_pred = model_age.predict(face_img, verbose=0)
                predicted_age = float(np.round(age_pred[0][0]))
                age_bin = get_age_bin(predicted_age)
                
                # Predict gender with error handling
                gender_pred = model_gender.predict(face_img, verbose=0)
                gender_arg = np.round(gender_pred).astype(np.uint8)
                gender = label_gender[gender_arg[0][0]]
                
                # Draw results on image
                cv2.putText(img, f'Age: {age_bin}, {gender}', 
                           (x - 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 255), 2, 
                           cv2.LINE_AA)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                continue
    
    return img

def process_video(uploaded_video):
    """Process uploaded video for age and gender detection."""
    try:
        # Save uploaded video to temporary file
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 120
        frame_time = 1.0 / fps
        
        # Calculate desired frame size (maintaining aspect ratio)
        max_width = 640
        if frame_width > max_width:
            scale_factor = max_width / frame_width
            new_width = int(frame_width * scale_factor)
            new_height = int(frame_height * scale_factor)
        else:
            new_width = frame_width
            new_height = frame_height

        model_age, model_gender = load_detection_models()
        label_gender = ['Male', 'Female']
        
        # Create a placeholder for the video feed
        frame_placeholder = st.empty()
        
        # Add a progress bar
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        # Initialize timing variables
        next_frame_time = time.time()

        while cap.isOpened():
            current_time = time.time()
            
            # Check if it's time to process the next frame
            if current_time < next_frame_time:
                time.sleep(0.001)  # Small sleep to prevent CPU overload
                continue
                
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            next_frame_time = current_time + frame_time
                
            # Update progress
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            progress_text.text(f'Processing frame {frame_count} of {total_frames} ({int(progress * 100)}%)')

            # Resize frame
            frame = cv2.resize(frame, (new_width, new_height))

            try:
                # Face detection with error handling
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        try:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            img_detect = cv2.resize(frame[y:y + h, x:x + w], dsize=(50, 50)).reshape(1, 50, 50, 3)
                            
                            # Predict age with error handling
                            predicted_age = float(np.round(model_age.predict(img_detect / 255., verbose=0))[0][0])
                            age_bin = str(get_age_bin(predicted_age))
                            
                            # Predict gender with error handling
                            gender_pred = model_gender.predict(img_detect / 255., verbose=0)
                            gender_arg = int(np.round(gender_pred)[0][0])
                            gender = str(label_gender[gender_arg])
                            
                            # Draw text with error handling
                            text = f'Age: {age_bin}, {gender}'
                            cv2.putText(frame, text, (x - 5, y - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                      (0, 255, 255), 2, cv2.LINE_AA)
                        except Exception as e:
                            st.error(f"Error processing detection: {str(e)}")
                            continue

            except Exception as e:
                st.error(f"Error in face detection: {str(e)}")
                continue

            # Convert BGR to RGB for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update the frame in Streamlit
            frame_placeholder.image(frame, channels="RGB", use_column_width=True)

            # Calculate processing delay and adjust timing
            processing_time = time.time() - current_time
            if processing_time < frame_time:
                time.sleep(frame_time - processing_time)

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        # Clean up temporary file
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)

def process_webcam():
    """Process webcam feed for age and gender detection."""
    try:
        # Add webcam selection
        webcam_options = {
            "Built-in Webcam": 0,
            "External Webcam": 1
        }
        
        # Add resolution options
        resolution_options = {
            "320x240": (320, 240),
            "640x480": (640, 480),
            "800x600": (800, 600),
            "1280x720": (1280, 720)
        }
        
        col1, col2 = st.columns(2)
        selected_webcam = col1.selectbox("Select Webcam", list(webcam_options.keys()))
        selected_resolution = col2.selectbox("Select Resolution", list(resolution_options.keys()))
        
        # Initialize session state for webcam
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
            
        # Add start/stop buttons in columns
        col3, col4 = st.columns(2)
        if col3.button('Start Webcam'):
            st.session_state.webcam_running = True
        if col4.button('Stop Webcam'):
            st.session_state.webcam_running = False
            
        # Create placeholder for webcam feed
        frame_placeholder = st.empty()
        
        if st.session_state.webcam_running:
            # Initialize webcam capture with selected device
            cap = cv2.VideoCapture(webcam_options[selected_webcam])
            
            if not cap.isOpened():
                st.error(f"Error: Could not open {selected_webcam}. Please check if the device is connected properly.")
                st.session_state.webcam_running = False
                return
            
            # Set resolution
            width, height = resolution_options[selected_resolution]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
            # Initialize face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            while st.session_state.webcam_running:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame")
                        break

                    # Flip the frame horizontally for a mirror effect
                    frame = cv2.flip(frame, 1)

                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.3,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    for (x, y, w, h) in faces:
                        # Expand the detection box
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Make box larger (2x width, 2.2x height)
                        new_w = int(w * 2.0)
                        new_h = int(h * 2.2)
                        
                        # Calculate new coordinates
                        new_x = center_x - new_w // 2
                        new_y = center_y - new_h // 2
                        
                        # Ensure coordinates are within frame
                        new_x = max(0, new_x)
                        new_y = max(0, new_y)
                        new_w = min(new_w, frame.shape[1] - new_x)
                        new_h = min(new_h, frame.shape[0] - new_y)
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
                        
                        # Extract face region
                        face_img = frame[new_y:new_y + new_h, new_x:new_x + new_w]
                        if face_img.size == 0:
                            continue
                            
                        # Process for age and gender
                        img_detect = cv2.resize(face_img, (50, 50))
                        img_detect = img_detect.reshape(1, 50, 50, 3) / 255.0
                        
                        # Predict age and gender
                        predicted_age = np.round(model_age.predict(img_detect))[0][0]
                        age_bin = get_age_bin(predicted_age)
                        
                        gender_pred = model_gender.predict(img_detect)
                        gender_arg = np.round(gender_pred).astype(np.uint8)
                        gender = label_gender[gender_arg[0][0]]
                        
                        # Draw text
                        text = f'Age: {age_bin}, {gender}'
                        cv2.putText(frame, text, (new_x, max(20, new_y - 10)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (0, 255, 255), 2, cv2.LINE_AA)

                    # Convert to RGB for Streamlit
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    
                    # Small sleep to prevent high CPU usage
                    time.sleep(0.01)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    continue
                
            # Release webcam when stopped
            cap.release()
            frame_placeholder.empty()
            
    except Exception as e:
        st.error(f"Error in webcam processing: {str(e)}")
        if 'cap' in locals():
            cap.release()

def main():
    """Main function for the Streamlit app."""
    
   
    # Header
    st.markdown("""
        <div class="header">
            <h1>Age and Gender Identification Through Neural Image Processing</h1>
            <p>Detect age and gender from images, videos, or webcam feed</p>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image", "🎥 Video", "📹 Webcam"])
    
    with tab1:
        st.subheader("Image Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key='image_uploader')
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            if img is not None:
                try:
                    processed_img = process_image(file_bytes)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Original Image")
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    with col2:
                        st.write("Processed Image")
                        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                        
                    st.success('Detection completed!')
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        st.subheader("Video Detection")
        uploaded_video = st.file_uploader("Upload a video file...", type=['mp4', 'avi', 'mov'], key='video_uploader')
        if uploaded_video is not None:
            st.write("Processing video...")
            process_video(uploaded_video)
    
    with tab3:
        st.subheader("Webcam Detection")
        
        # Webcam controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            webcam_options = {
                "Built-in Webcam": 0,
                "External Webcam": 1
            }
            selected_webcam = st.selectbox(
                "Select Camera",
                list(webcam_options.keys())
            )
        
        with col2:
            st.write("")  # Add vertical space
            st.write("")  # Add more space to align with dropdown
            if st.button('Start', use_container_width=True):
                st.session_state.webcam_running = True
        
        with col3:
            st.write("")  # Add vertical space
            st.write("")  # Add more space to align with dropdown
            if st.button('Stop', use_container_width=True):
                st.session_state.webcam_running = False

        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        
        # Status indicator
        if st.session_state.webcam_running:
            st.markdown('**Status:** Running', unsafe_allow_html=True)
        else:
            st.markdown('**Status:** Stopped', unsafe_allow_html=True)

        # Create a placeholder for the webcam feed
        frame_placeholder = st.empty()
        
        # Process webcam feed
        if st.session_state.webcam_running:
            try:
                cap = cv2.VideoCapture(webcam_options[selected_webcam])
                
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FPS, 15)  
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  
                
                # Initialize face detection and models
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                model_age, model_gender = load_detection_models()
                label_gender = ['Male', 'Female']
                
                # Frame timing control
                frame_time = 1/15  
                prev_frame_time = time.time()
                frame_count = 0
                
                while st.session_state.webcam_running:
                    try:
                        current_time = time.time()
                        
                        # Process every 3rd frame
                        frame_count += 1
                        if frame_count % 3 != 0:
                            ret, _ = cap.read()  
                            continue

                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture frame")
                            break

                        # Convert frame to RGB for face detection (smaller size)
                        frame_small = cv2.resize(frame, (320, 240))
                        rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                        
                        # Detect faces using tiny model
                        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                        
                        if len(faces) > 0:
                            for (x, y, w, h) in faces:
                                try:
                                    # Scale coordinates back to original size
                                    scale_x = frame.shape[1] / 320
                                    scale_y = frame.shape[0] / 240
                                    x_orig = int(x * scale_x)
                                    y_orig = int(y * scale_y)
                                    w_orig = int(w * scale_x)
                                    h_orig = int(h * scale_y)
                                    
                                    # Extract and process face region
                                    face_img = frame[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]
                                    if face_img.size == 0:
                                        continue

                                    img_detect = cv2.resize(face_img, dsize=(50, 50)).reshape(1, 50, 50, 3)
                                    
                                    # Detect Age
                                    predicted_age = np.round(model_age.predict(img_detect / 255.))[0][0]
                                    age_bin = str(get_age_bin(predicted_age))
                                    
                                    # Detect Gender
                                    gender_arg = np.round(model_gender.predict(img_detect / 255.)).astype(np.uint8)
                                    gender = str(label_gender[gender_arg[0][0]])
                                    
                                    # Draw text
                                    text = f'Age: {age_bin}, {gender}'
                                    cv2.putText(frame, text, 
                                              (x_orig, max(20, y_orig - 10)), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                              (0, 255, 255), 2, cv2.LINE_AA)

                                    # Draw rectangle
                                    cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (0, 255, 0), 2)
                                except Exception as e:
                                    st.error(f"Error processing detection: {str(e)}")
                                    continue

                        # Display FPS
                        fps = 1/(current_time - prev_frame_time)
                        cv2.putText(frame, f'FPS: {int(fps)}', (10, 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (0, 255, 0), 1)
                        
                        # Display frame
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                        prev_frame_time = current_time
                        
                    except Exception as e:
                        continue
                    
            except Exception as e:
                st.error(f"Error in webcam processing: {str(e)}")
            finally:
                cap.release()
                frame_placeholder.empty()

    # Add custom footer
    st.markdown("""
        <div class="custom-footer">
            Made with <span>♥</span> by Tanmay Hadke
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
