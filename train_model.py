import cv2
import numpy as np
import glob
import os
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
# --- NEW IMPORT ---
from sklearn.metrics import classification_report

# --- 1. Define Constants ---
TARGET_W, TARGET_H = 800, 600
TARGET_ASPECT_RATIO = TARGET_W / TARGET_H
GRID_ROWS, GRID_COLS = 8, 8
GRID_W = TARGET_W // GRID_COLS  # 100
GRID_H = TARGET_H // GRID_ROWS  # 75
IMAGE_DIR = "images"
LABEL_DIR = "labels"

# --- 2. Copy Your process_image Helper Function ---
def process_image(path: str) -> np.ndarray | None:
    """
    Loads an image, enforces 4:3 aspect ratio by center-cropping,
    and resizes to exactly 800x600.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read image at {path}")
        return None
    h, w, _ = img.shape
    current_aspect_ratio = w / h
    if not np.isclose(current_aspect_ratio, TARGET_ASPECT_RATIO):
        if current_aspect_ratio > TARGET_ASPECT_RATIO: 
            new_w = int(TARGET_ASPECT_RATIO * h)
            x_start = (w - new_w) // 2
            img = img[:, x_start:x_start + new_w]
        else: 
            new_h = int(w / TARGET_ASPECT_RATIO)
            y_start = (h - new_h) // 2
            img = img[y_start:y_start + new_h, :]
    h, w, _ = img.shape
    if w > TARGET_W:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=interpolation)
    return img

# --- 3. Feature Extraction and Data Loading ---
def extract_cell_features(cell_image: np.ndarray) -> np.ndarray:
    """
    Extracts HOG features from a single 100x75 cell.
    Converts to grayscale first, as HOG requires it.
    """
    gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    
    features = hog(gray_cell, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', 
                   visualize=False, transform_sqrt=True)
    return features

def load_data():
    """
    Loads all labeled data, extracting HOG features and labels
    for every single grid cell.
    """
    print("Loading labeled data and extracting features...")
    X_features = []  # To store HOG feature vectors
    y_labels = []    # To store labels (0 or 1)
    
    label_paths = glob.glob(os.path.join(LABEL_DIR, "*.npy"))
    if not label_paths:
        return None, None

    for label_path in label_paths:
        basename = os.path.basename(label_path).replace(".npy", "")
        img_path = None
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            potential_path = glob.glob(os.path.join(IMAGE_DIR, f"{basename}{ext[1:]}"))
            if potential_path:
                img_path = potential_path[0]
                break
        
        if not img_path:
            continue

        img = process_image(img_path)
        if img is None:
            continue
            
        grid_map = np.load(label_path) # The 8x8 label map
        
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                y1, x1 = i * GRID_H, j * GRID_W
                y2, x2 = y1 + GRID_H, x1 + GRID_W
                cell_patch = img[y1:y2, x1:x2]
                
                features = extract_cell_features(cell_patch)
                
                X_features.append(features)
                y_labels.append(grid_map[i, j])

    if not X_features:
        return None, None

    print(f"Loaded {len(y_labels)} cell examples from {len(label_paths)} images.")
    return np.array(X_features), np.array(y_labels)

# --- 4. Main Training Logic ---
if __name__ == "__main__":
    X_train, y_train = load_data()
    
    if X_train is None:
        print("Error: No labels found in the 'labels' directory.")
        print("Please label 10-20 images with `wildlife_labeler.py` first.")
    else:
        # --- 1. Scale the features ---
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # --- 2. Train the SVM ---
        print("Training SVM model...")
        model = SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced')
        
        model.fit(X_train_scaled, y_train)
        
        print("Training complete.")
        
        # --- 3. Evaluate Model and Save Output --- (THIS IS THE NEW SECTION)
        print("\n--- Evaluating model on training data ---")
        
        # Use the trained model to predict on the same data
        y_pred = model.predict(X_train_scaled)
        
        # Generate the text report
        report = classification_report(y_train, y_pred, target_names=["No Animal (0)", "Animal (1)"], zero_division=0)
        
        print(report)
        
        # Save the report to a file
        OUTPUT_FILE = "model_training_report.txt"
        try:
            with open(OUTPUT_FILE, 'w') as f:
                f.write("--- SVM Model Training Report ---\n\n")
                f.write(f"Total cell examples trained on: {len(y_train)}\n")
                f.write(f"Positive (Animal) samples: {np.sum(y_train)}\n")
                f.write(f"Negative (No Animal) samples: {len(y_train) - np.sum(y_train)}\n\n")
                f.write("--- Classification Report ---\n")
                f.write(report)
            print(f"--- Model output report saved to {OUTPUT_FILE} ---")
        except IOError as e:
            print(f"Warning: Could not save report. {e}")

        # --- 4. Save the model AND the scaler ---
        print("\n--- Saving model files ---")
        joblib.dump(model, "wildlife_svm.joblib")
        joblib.dump(scaler, "scaler.joblib")
        
        print("--- Model saved as wildlife_svm.joblib ---")
        print("--- Scaler saved as scaler.joblib ---")