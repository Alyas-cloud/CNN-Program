import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\Rapid Computer's\Desktop\cnn fruits\fruit_classifier.h5")
print("Model loaded successfully.")

# Define class labels (ensure they match your training data order)
class_labels = ['apple', 'grape', 'mango', 'orange', 'watermelon']  # Adjust as needed

# Prediction function
def predict_image(img_path):
    """
    Predict the class of a fruit image.
    Args:
        img_path (str): Path to the image file.
    Returns:
        tuple: Predicted class label and confidence score.
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))  # Match input size of the model
    img_array = image.img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Predict with the model
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)  # Index of highest confidence
    confidence = predictions[0][predicted_class_index]  # Confidence of prediction

    # Display the result
    print(f"Predicted class: {class_labels[predicted_class_index]} with confidence {confidence:.2f}")
    return class_labels[predicted_class_index], confidence

# Visualization function
def visualize_prediction(img_path):
    """
    Display the image and its predicted class with confidence.
    Args:
        img_path (str): Path to the image file.
    """
    # Load the image
    img = image.load_img(img_path, target_size=(128, 128))
    plt.imshow(img)
    plt.axis('off')

    # Predict and display the result
    predicted_class, confidence = predict_image(img_path)
    plt.title(f"{predicted_class} ({confidence*100:.2f}% confidence)")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Path to the test image
    test_image_path = r"C:\Users\Rapid Computer's\Desktop\cnn fruits\dataset\train\mango\mango2.jpg"  # Replace with your image file path

    # Predict and visualize
    visualize_prediction(test_image_path)
