import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Paths to dataset directories
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Step 1: Data Preparation
# Augment the training data to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values
    shear_range=0.2,           # Random shearing
    zoom_range=0.2,            # Random zoom
    rotation_range=30,         # Random rotation
    width_shift_range=0.2,     # Horizontal shift
    height_shift_range=0.2,    # Vertical shift
    horizontal_flip=True       # Random horizontal flip
)

# Validation data should only be rescaled (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),    # Resize all images to 128x128
    batch_size=32,
    class_mode='categorical'   # Multi-class classification
)

validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Step 2: Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # First convolution layer
    MaxPooling2D(pool_size=(2, 2)),                                   # First pooling layer
    
    Conv2D(64, (3, 3), activation='relu'),                            # Second convolution layer
    MaxPooling2D(pool_size=(2, 2)),                                   # Second pooling layer
    
    Conv2D(128, (3, 3), activation='relu'),                           # Third convolution layer
    MaxPooling2D(pool_size=(2, 2)),                                   # Third pooling layer
    
    Flatten(),                                                        # Flatten the 3D feature maps
    Dense(128, activation='relu'),                                    # Fully connected layer
    Dropout(0.5),                                                     # Dropout for regularization
    Dense(train_data.num_classes, activation='softmax')               # Output layer for multi-class
])

# Step 3: Compile the Model
model.compile(
    optimizer='adam',                 # Adam optimizer
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']              # Track accuracy
)

# Step 4: Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Early stopping

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=100,                        # Number of epochs
    callbacks=[early_stopping]
)

# Step 5: Save the Model
model.save('fruit_classifier.h5')
print("Model saved as 'fruit_classifier.h5'")

# Step 6: Visualize Training and Validation Metrics
# Accuracy plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()
