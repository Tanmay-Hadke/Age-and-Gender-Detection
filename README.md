# Age and Gender Detection Through Neural Image Processing  

## Abstract  
This project focuses on developing a system to identify age and gender from facial images. Utilizing state-of-the-art neural networks and diverse preprocessing techniques, we explore feature extraction methods, including OpenCV's resizing function and Pillow, to build robust models. The project serves as a foundational application in computer vision, enabling advancements in analytics, security, and user experience domains.

---

## Problem Statement  
Accurate detection of age and gender is crucial in various industries such as retail, security, and healthcare. Despite advancements in machine learning, challenges persist due to variations in lighting, pose, and diverse facial features. This project addresses the problem by employing neural image processing techniques to enhance model accuracy and robustness.

---

## Approach  
The project is divided into three primary phases:  

1. **Data Preprocessing**  
   - Implemented in the `preprocessing.ipynb` notebook.
   - Feature extraction methods:
     - OpenCV's `resize` function to standardize image dimensions.
     - Pillow library for alternate resizing and feature manipulation.
   - Data augmentation to address class imbalances and overfitting.

2. **Model Training**  
   - Conducted in `train_models.ipynb`.
   - Two distinct neural network architectures are tested:
     - Baseline CNN with fine-tuned parameters.
     - Pre-trained deep learning models for transfer learning.
   - Metrics used: Accuracy, precision, recall, and F1-score.

3. **Detection and Testing**  
   - The `detect.py` script implements the trained model for real-time detection.
   - Input: Image/video streams.
   - Output: Predicted age and gender displayed with bounding boxes.

---

## Results  
- Model Accuracy: Achieved **high precision** and **recall** on benchmark datasets.
- Feature extraction using OpenCV demonstrated superior performance in speed.
- Data augmentation significantly reduced overfitting, enabling generalization across unseen data.

---

## Installation and Usage  

### Prerequisites  
Ensure you have the following installed:  
- Python 3.x  
- OpenCV  
- TensorFlow/Keras  
- Pillow  

### Steps to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Tanmay-Hadke/Age-and-Gender-Detection/
   cd Age-and-Gender-Detection
   ```  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Run Preprocessing:  
   Execute `preprocessing.ipynb` to prepare the dataset.  

4. Train Models:  
   Run `train_models.ipynb` to train and evaluate the model.  

5. Detection:  
   Execute the `detect.py` script for real-time predictions:  
   ```bash
   python detect.py
   ```

---

## Future Scope  
- Enhance the dataset diversity to improve model robustness.
- Explore other feature extraction techniques like histogram equalization.
- Incorporate explainability techniques to interpret model predictions.

---
