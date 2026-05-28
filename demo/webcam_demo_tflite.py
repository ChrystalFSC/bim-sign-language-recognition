"""
Real-time Sign Language Recognition with MediaPipe & Deployed TFLite Model
Uses webcam → detects hand landmarks → crops ROI → runs TFLite inference
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import mediapipe as mp

# === SETTINGS ===
TFLITE_PATH = "deployment/mobilenetv3_small_float16.tflite"
LABEL_PATH = "deployment/label_map.txt"

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_hand_bbox(hand_landmarks, frame_width, frame_height, padding=40):
    """Extract tight hand ROI bounding box with safety padding."""
    x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
    y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]
    
    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = min(frame_width, int(max(x_coords)) + padding)
    y_max = min(frame_height, int(max(y_coords)) + padding)
    
    return x_min, y_min, x_max, y_max

def main():
    print("="*60)
    print("REAL-TIME TFLITE SIGN LANGUAGE RECOGNITION (UK English)")
    print("="*60)
    
    # 1. Load labels
    if not os.path.exists(LABEL_PATH):
        print(f"[ERROR] Label map not found at {LABEL_PATH}")
        return
    with open(LABEL_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(classes)} classes from label map.")
    
    # 2. Load TFLite Model Interpreter
    if not os.path.exists(TFLITE_PATH):
        print(f"[ERROR] TFLite deployment model not found at {TFLITE_PATH}")
        return
    print(f"Loading optimised TFLite model from: {TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    
    # Get model parameters
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    print(f"✓ Deployed model loaded successfully!")
    print(f"  Input Shape: {input_shape}")
    print(f"  Output Classes: {len(classes)}")
    
    print("\nStarting webcam capture...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    # Instantiate MediaPipe tracker
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
        
        frame = cv2.flip(frame, 1)  # Mirror frame
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw MediaPipe hand skeleton overlay
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract hand bounding box
                x1, y1, x2, y2 = get_hand_bbox(
                    hand_landmarks, frame_width, frame_height)
                
                # Draw tracking ROI rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Crop and preprocess ROI
                hand_crop = rgb_frame[y1:y2, x1:x2]
                
                if hand_crop.size > 0:
                    # Resize to model input specification (224x224)
                    hand_resized = cv2.resize(hand_crop, (224, 224))
                    
                    # Normalise to float32 input (0.0 to 255.0) matching training rescaler
                    hand_array = hand_resized.astype(np.float32)
                    hand_array = np.expand_dims(hand_array, axis=0)
                    
                    # Run TFLite inference
                    interpreter.set_tensor(input_details[0]['index'], hand_array)
                    interpreter.invoke()
                    
                    # Retrieve predictions
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    pred_idx = np.argmax(output_data)
                    confidence = output_data[pred_idx]
                    prediction_text = classes[pred_idx]
        
        # Draw status panel text on frame
        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.putText(frame, f"Sign: {prediction_text}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("BIM Sign Recognition (TFLite)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("screenshot.png", frame)
            print("Screenshot saved to screenshot.png!")
            
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nInference stopped.")

if __name__ == "__main__":
    main()
