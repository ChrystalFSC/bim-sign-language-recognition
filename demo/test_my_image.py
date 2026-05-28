"""
Test the trained model with your own images
Usage: python test_my_image.py path/to/your/image.jpg
"""
import sys
import os
import numpy as np
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

# Model path - update if needed
MODEL_PATH = "output/best_model_stage3_1.keras"

# Class names
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for loading model."""
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return focal_loss_fn

def load_model():
    """Load the trained model."""
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'focal_loss_fn': focal_loss(5.0, 0.25)}
    )
    print("Model loaded!")
    return model

def preprocess_image(image_path):
    """Load and preprocess an image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def predict(model, image_path):
    """Make prediction on an image."""
    img_array, original_img = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    
    # Get top 5 predictions
    top5_indices = np.argsort(predictions[0])[-5:][::-1]
    
    print("\n" + "="*50)
    print(f"Image: {image_path}")
    print("="*50)
    
    print(f"\n🏆 PREDICTION: {CLASSES[top5_indices[0]]} ({predictions[0][top5_indices[0]]*100:.1f}% confidence)")
    
    print("\nTop 5 Predictions:")
    for i, idx in enumerate(top5_indices):
        confidence = predictions[0][idx] * 100
        bar = "█" * int(confidence / 5)
        print(f"  {i+1}. {CLASSES[idx]}: {confidence:5.1f}% {bar}")
    
    return CLASSES[top5_indices[0]], predictions[0][top5_indices[0]]

def main():
    if len(sys.argv) < 2:
        print("\n" + "="*50)
        print("MANUAL IMAGE TESTING")
        print("="*50)
        print("\nUsage:")
        print("  python test_my_image.py <image_path>")
        print("\nExamples:")
        print("  python test_my_image.py my_test.jpg")
        print("  python test_my_image.py C:/path/to/image.png")
        print("  python test_my_image.py test_data/A/1.jpg")
        print("\n" + "="*50)
        
        # Interactive mode
        model = load_model()
        
        while True:
            print("\n")
            image_path = input("Enter image path (or 'q' to quit): ").strip()
            
            if image_path.lower() == 'q':
                break
            
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                continue
            
            try:
                predict(model, image_path)
            except Exception as e:
                print(f"❌ Error: {e}")
    else:
        # Command line mode
        image_path = sys.argv[1]
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            sys.exit(1)
        
        model = load_model()
        predict(model, image_path)

if __name__ == "__main__":
    main()
