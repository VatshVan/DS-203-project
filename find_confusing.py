import cv2
import numpy as np
import glob
import os
import joblib
from skimage.feature import hog

# --- 1. Define Constants (Must match your labeler) ---
TARGET_W, TARGET_H = 800, 600
TARGET_ASPECT_RATIO = TARGET_W / TARGET_H
GRID_ROWS, GRID_COLS = 8, 8
GRID_W = TARGET_W // GRID_COLS  # 100
GRID_H = TARGET_H // GRID_ROWS  # 75
IMAGE_DIR = "images"
LABEL_DIR = "labels"

# --- How many images to find? ---
N_IMAGES_TO_FIND = 10 # You asked for 5 up to 10.

# --- 2. Copy Helper Functions (process_image & extract_cell_features) ---
def process_image(path: str) -> np.ndarray | None:
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

def extract_cell_features(cell_image: np.ndarray) -> np.ndarray:
    gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_cell, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', 
                   visualize=False, transform_sqrt=True)
    return features

# --- 3. Main Finding Logic ---
if __name__ == "__main__":
    print("--- Finding Most Confusing Images (SVM) ---")
    
    # 1. Load the trained model and scaler
    if not os.path.exists("wildlife_svm.joblib") or not os.path.exists("scaler.joblib"):
        print("Error: `wildlife_svm.joblib` or `scaler.joblib` not found.")
        print("Please run `train_model.py` first.")
        exit()
        
    model = joblib.load("wildlife_svm.joblib")
    scaler = joblib.load("scaler.joblib")
    
    # 2. Find all UNLABELED images
    labeled_files = os.listdir(LABEL_DIR)
    labeled_basenames = set([os.path.splitext(f)[0] for f in labeled_files])
    
    all_image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + \
                      glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) + \
                      glob.glob(os.path.join(IMAGE_DIR, "*.png"))
                      
    unlabeled_paths = []
    for path in all_image_paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        if basename not in labeled_basenames:
            unlabeled_paths.append(path)
            
    if not unlabeled_paths:
        print("Congratulations! All images have been labeled.")
        exit()

    print(f"Found {len(unlabeled_paths)} unlabeled images to score...")
    
    # 3. Score all unlabeled images for uncertainty
    uncertainty_scores = []
    
    for path in unlabeled_paths:
        img = process_image(path)
        if img is None:
            continue
            
        # Get features for all 64 cells in this image
        image_cell_features = []
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                y1, x1 = i * GRID_H, j * GRID_W
                y2, x2 = y1 + GRID_H, x1 + GRID_W
                cell_patch = img[y1:y2, x1:x2]
                features = extract_cell_features(cell_patch)
                image_cell_features.append(features)
        
        X_cells = np.array(image_cell_features)
        
        # Scale the features using the *loaded* scaler
        try:
            X_cells_scaled = scaler.transform(X_cells)
        except Exception as e:
            print("Error: Scaler is not fitted. This shouldn't happen. Exiting.")
            exit()
        
        # Get probability scores (requires model to be
        # trained with probability=True)
        # This will be a (64, 2) array: [P(class 0), P(class 1)]
        probas = model.predict_proba(X_cells_scaled)
        
        # Get the probability of the positive class (class 1)
        probas_class_1 = probas[:, 1]
        
        # Calculate uncertainty:
        # We want scores closest to 0.5 (max confusion).
        # abs(P - 0.5) will be LOWEST for uncertain cells.
        uncertainty = np.abs(probas_class_1 - 0.5)
        
        # The score for the *image* is its average cell uncertainty.
        # We want the image with the LOWEST average score (closest to 0.5)
        image_score = np.mean(uncertainty)
        
        uncertainty_scores.append((image_score, path))

    # 4. Sort and find the Top N
    
    # Sort by score (ascending). The lowest scores are the most uncertain.
    uncertainty_scores.sort(key=lambda x: x[0])
    
    top_n_to_label = uncertainty_scores[:N_IMAGES_TO_FIND]
    
    # --- THIS IS THE NEW PART ---
    ACTIVE_LEARNING_LIST = "to_label.txt"
    
    print(f"\n--- Top {N_IMAGES_TO_FIND} Images to Label Next ---")
    print(f"--- Saving list to '{ACTIVE_LEARNING_LIST}' ---")

    try:
        with open(ACTIVE_LEARNING_LIST, 'w') as f:
            for score, path in top_n_to_label:
                print(f"File: {os.path.basename(path)} (Uncertainty Score: {score:.4f})")
                f.write(f"{path}\n") # Write the full path to the file
                
        print(f"\nRun `wildlife_labeler.py` to label these {N_IMAGES_TO_FIND} files.")
        
    except IOError as e:
        print(f"Error: Could not write to file {ACTIVE_LEARNING_LIST}. {e}")