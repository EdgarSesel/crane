import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Group containers into rows based on similar y-coordinates
def group_containers_by_rows(container_rects, tolerance=20):
    container_rects = sorted(container_rects, key=lambda x: x[1])  # Sort by y-coordinate
    rows = []
    current_row = [container_rects[0]]
    
    for rect in container_rects[1:]:
        if abs(rect[1] - current_row[0][1]) <= tolerance:
            current_row.append(rect)
        else:
            rows.append(current_row)
            current_row = [rect]
    
    if current_row:
        rows.append(current_row)
    
    return rows

# Function to detect containers and measure gaps
def detect_container_gaps(image_path):
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    
    # Find contours (edges of containers)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Store the bounding rectangles of containers
    container_rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 18 and h > 18:  # Filter based on minimum size (tune as needed)
            container_rects.append((x, y, w, h))

    # Group containers into rows
    rows = group_containers_by_rows(container_rects)
    
    all_gaps = []  # List to store gaps between all containers
    
    # Calculate gaps between containers within each row
    for row in rows:
        row = sorted(row, key=lambda x: x[0])  # Sort containers in the row by x-coordinate
        row_gaps = []
        for i in range(1, len(row)):
            gap = row[i][0] - (row[i-1][0] + row[i-1][2])
            if gap > 0:  # Only count positive gaps
                row_gaps.append(gap)
        
        if row_gaps:
            all_gaps.extend(row_gaps)  # Add gaps from this row to the total list
    
    # Return the average gap size of all containers
    if all_gaps:
        avg_gap = np.mean(all_gaps)
        return avg_gap
    return 0  # Return 0 if no valid gaps are found

# Example dataset of images with labels (1 = Straddle, 0 = RMG)
images = ['1.png', '2.png', '3.png', "4.png", "5.png", "6.png", "7.png", "8.png", "9.png", "13.png", "14.png"]  # Add paths to your images
labels = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1]  # Labels corresponding to each image, 1 = Straddle, 0 = RMG

# Extract features (average gap size for each image)
features = [detect_container_gaps(img) for img in images]

# Train a simple classifier on the images
X_train = np.array(features).reshape(-1, 1)  # Reshape to match sklearn's expected input
y_train = labels  # The corresponding labels (1 for Straddle, 0 for RMG)

clf = DecisionTreeClassifier()  # Initialize Decision Tree Classifier
clf.fit(X_train, y_train)  # Train the model using the gap features

# Test the classifier on a new image (port_image9.png)
new_image = '15.png'
new_image_gap = detect_container_gaps(new_image)  # Calculate the average gap size for the new image

# Predict whether it's a Straddle Carrier (1) or RMG Crane (0)
classification = clf.predict([[new_image_gap]])

# Output the result
if classification == 1:
    print(f"Straddle Carriers detected in {new_image} (Average gap: {new_image_gap} pixels)")
else:
    print(f"RMG Crane detected in {new_image} (Average gap: {new_image_gap} pixels)")
