"""
Real-time Sign Language Recognition with MediaPipe Hand Detection
Uses webcam → detects hand → crops → classifies

Install first:
    pip install mediapipe opencv-python
"""
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp

# === SETTINGS ===
MODEL_PATH = "output/best_model_stage3_1.keras"
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return focal_loss_fn

def get_hand_bbox(hand_landmarks, frame_width, frame_height, padding=50):
    """Get bounding box from hand landmarks with padding."""
    x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
    y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]
    
    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = min(frame_width, int(max(x_coords)) + padding)
    y_max = min(frame_height, int(max(y_coords)) + padding)
    
    return x_min, y_min, x_max, y_max

def main():
    print("="*60)
    print("REAL-TIME SIGN LANGUAGE RECOGNITION")
    print("="*60)
    print("\nLoading model...")
    
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'focal_loss_fn': focal_loss(5.0, 0.25)}
    )
    print("Model loaded!")
    
    print("\nStarting webcam...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    prediction_text = "Show your hand..."
    confidence = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get bounding box
                x1, y1, x2, y2 = get_hand_bbox(
                    hand_landmarks, frame_width, frame_height)
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Crop and preprocess
                hand_crop = rgb_frame[y1:y2, x1:x2]
                
                if hand_crop.size > 0:
                    # Resize to model input
                    hand_resized = cv2.resize(hand_crop, (224, 224))
                    hand_array = hand_resized / 255.0
                    hand_array = np.expand_dims(hand_array, axis=0)
                    
                    # Predict
                    predictions = model.predict(hand_array, verbose=0)
                    pred_idx = np.argmax(predictions[0])
                    confidence = predictions[0][pred_idx]
                    prediction_text = CLASSES[pred_idx]
        
        # Display prediction
        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.putText(frame, f"Sign: {prediction_text}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Sign Language Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("screenshot.png", frame)
            print("Screenshot saved!")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
