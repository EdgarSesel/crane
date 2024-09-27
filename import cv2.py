import cv2
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def load_data(image_dir, labels_file, target_size=(256, 256)):
    images = []
    labels = []
    # Read labels from CSV file
    labels_df = pd.read_csv(labels_file)
    for _, row in labels_df.iterrows():
        image_name = row['filename']
        label = row['label']
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            edges = preprocess_image(image_path, target_size)
            images.append(edges)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    return images, labels

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
image_dir = '.'  # Current directory
labels_file = 'labels.csv'  # Path to the CSV file with labels
images, labels = load_data(image_dir, labels_file)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow(X_train, y_train, batch_size=32)
val_generator = datagen.flow(X_val, y_val, batch_size=32)

# Create and train the model
model = create_model()
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the model
model.save('container_classifier_model.h5')

# Example prediction
def classify_image(image_path, model):
    edges = preprocess_image(image_path)
    edges = edges.reshape(1, edges.shape[0], edges.shape[1], 1) / 255.0
    prediction = model.predict(edges)
    return 'RMG' if prediction < 0.5 else 'Straddle'

# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('container_classifier_model.h5')

# Classify a new image
image_path = '17.png'  # Example image
classification = classify_image(image_path, model)
print(f'The image is classified as: {classification}')