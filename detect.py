import cv2
import numpy as np
from keras.layers import DepthwiseConv2D
from keras.models import load_model
from keras.losses import MeanSquaredError
from yoloface import face_analysis
import tkinter as tk
from tkinter import filedialog

# Load model
model_age = load_model('best_model.keras')
model_gender = load_model('best_model_gender.keras')

# Gender labels
label_gender = ['Male', 'Female']

# Function to categorize predicted age into specific bins
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

# Function to detect age and gender
def detect_image(url):
    img = cv2.imread(url)
    # Face detection box
    face = face_analysis()
    _, box, _ = face.face_detection(image_path=url, model='full')
    for x, y, w, h in box:
        cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
        # Rescale the face region before prediction
        img_detect = cv2.resize(img[y:y+w, x:x+h], dsize=(50, 50)).reshape(1, 50, 50, 3)
        # Detect Age
        predicted_age = np.round(model_age.predict(img_detect / 255.))[0][0]
        age_bin = get_age_bin(predicted_age)
        # Detect Gender
        gender_arg = np.round(model_gender.predict(img_detect / 255.)).astype(np.uint8)
        gender = label_gender[gender_arg[0][0]]
        # Draw
        cv2.putText(img, f'Age: {age_bin}, {gender}', (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    # Note: Removed resizing of the output image to maintain original dimensions
    cv2.imshow('detect', img)
    cv2.waitKey(0)
    return img

def rescaleFrame(frame, scale=0.25):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dims = (width, height)
    return cv2.resize(frame, dims, interpolation=cv2.INTER_AREA)

def select_image_file():
    """Open a file dialog to select an image file."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.update()  # Ensure the window is initialized
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

def detect_from_webcam():
    """Capture video from the webcam and detect faces."""
    cap = cv2.VideoCapture(0)
    face = face_analysis()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Face detection
        _, box, _ = face.face_detection(frame_arr=frame, frame_status=True, model='tiny')

        for x, y, w, h in box:
            cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)
            img_detect = cv2.resize(frame[y:y + w, x:x + h], dsize=(50, 50)).reshape(1, 50, 50, 3)
            # Detect Age
            predicted_age = np.round(model_age.predict(img_detect / 255.))[0][0]
            age_bin = get_age_bin(predicted_age)
            # Detect Gender
            gender_arg = np.round(model_gender.predict(img_detect / 255.)).astype(np.uint8)
            gender = label_gender[gender_arg[0][0]]
            # Draw
            cv2.putText(frame, f'Age: {age_bin}, {gender}', (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow('Webcam Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
image_path = select_image_file()
if image_path:
    detect_image(image_path)
else:
    print("No image selected.")
    detect_from_webcam()
    detect_from_webcam()
