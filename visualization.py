import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to focus on a single row of containers
    height, width = gray.shape
    row_height = height // 10  # Assuming each row is roughly 1/10th of the image height
    resized = cv2.resize(gray[row_height:2*row_height, :], target_size)
    
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def classify_image(image_path, model):
    edges = preprocess_image(image_path)
    edges = edges.reshape(1, edges.shape[0], edges.shape[1], 1) / 255.0
    prediction = model.predict(edges)
    return 'RMG' if prediction < 0.5 else 'Straddle'

def visualize_single_image(image_path, model):
    edges = preprocess_image(image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.show()

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((256, 256), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        panel.config(image=photo)
        panel.image = photo

        # Classify the image
        classification = classify_image(file_path, model)
        result_text.set(f'Classification: {classification}')

# Load the trained model
model = load_model('container_classifier_model.h5')

# Create the main window
root = tk.Tk()
root.title("Image Classifier")
root.geometry("400x400")

# Create a panel to display the image
panel = tk.Label(root)
panel.pack()

# Create a button to open an image
btn = tk.Button(root, text="Open Image", command=open_image)
btn.pack()

# Create a label to display the classification result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text)
result_label.pack()

# Start the GUI event loop
root.mainloop()
