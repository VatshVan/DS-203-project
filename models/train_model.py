import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import joblib
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# --- Configuration ---
GRID_W = 100
GRID_H = 75
TARGET_ASPECT_RATIO = 4.0 / 3.0
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
MODEL_PATH = "wildlife_detector_model.pkl"
TRAINING_IMAGE_FOLDER = "images" # <-- Folder with *only normal* images

# --- Helper Functions (Copied from your script) ---

def process_image_for_training(path: str) -> np.ndarray | None:
    """Loads and processes an image using the same rules as training."""
    img = cv2.imread(path)
    if img is None: return None
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
    if w > 800 or h > 600:
        img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
    h, w, _ = img.shape
    if w < 800 or h < 600: return None
    return img

def extract_features_from_image(img: np.ndarray) -> np.ndarray:
    """Extracts features from all 64 cells in a single image."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features_list = []
    for y in range(0, gray_img.shape[0], GRID_H):
        for x in range(0, gray_img.shape[1], GRID_W):
            cell = gray_img[y:y+GRID_H, x:x+GRID_W]
            cell_features = []
            cell_features.append(np.mean(cell))
            cell_features.append(np.std(cell))
            lbp = local_binary_pattern(cell, LBP_POINTS, LBP_RADIUS, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), density=True)
            cell_features.extend(hist)
            edges = cv2.Canny(cell, 50, 150)
            cell_features.append(np.sum(edges > 0) / cell.size)
            features_list.append(np.array(cell_features))
    return np.vstack(features_list)

# --- Main Training Execution ---
if __name__ == "__main__":
    
    # 1. Select the model you want to train
    # Options: 'iso_forest', 'one_class_svm', 'lof'
    CHOSEN_MODEL_NAME = 'iso_forest' 
    
    # 2. Find all training images
    image_paths = glob.glob(os.path.join(TRAINING_IMAGE_FOLDER, "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(TRAINING_IMAGE_FOLDER, "*.png")))
    
    if not image_paths:
        print(f"Error: No images found in '{TRAINING_IMAGE_FOLDER}'.")
        print("Please create this folder and add 'normal' (no wildlife) images to it.")
    else:
        print(f"Found {len(image_paths)} training images. Extracting features...")
        
        # 3. Extract features from all training images
        all_features = []
        for path in image_paths:
            img = process_image_for_training(path)
            if img is not None:
                features = extract_features_from_image(img)
                all_features.append(features)
            else:
                print(f"Skipping {path} (size/aspect issue)")
        
        if not all_features:
            print("No valid training images were processed. Exiting.")
            exit()
            
        X_train = np.vstack(all_features)
        print(f"Total features extracted for training: {X_train.shape[0]}")
        
        # 4. Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 5. Define and train the chosen model
        model = None
        
        if CHOSEN_MODEL_NAME == 'iso_forest':
            print("Training IsolationForest...")
            # contamination: The expected proportion of anomalies in the *training* data.
            # Since we assume training data is "clean", we set this very low.
            model = IsolationForest(contamination=0.001, random_state=42, n_estimators=100)
        
        elif CHOSEN_MODEL_NAME == 'one_class_svm':
            print("Training OneClassSVM...")
            # nu: An upper bound on the fraction of training errors and a lower
            # bound of the fraction of support vectors. (Similar to contamination)
            model = OneClassSVM(nu=0.001, kernel="rbf", gamma="auto")
            
        elif CHOSEN_MODEL_NAME == 'lof':
            print("Training LocalOutlierFactor...")
            # novelty=True: This is crucial! It allows the LOF model to be
            # used for predicting on *new*, unseen data.
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.001, novelty=True)
        
        else:
            print(f"Error: Unknown model '{CHOSEN_MODEL_NAME}'.")
            exit()

        model.fit(X_train_scaled)
        
        # 6. Save the scaler and the trained model
        print(f"Training complete. Saving model to '{MODEL_PATH}'...")
        model_data = {
            'scaler': scaler,
            'model': model,
            'model_name': CHOSEN_MODEL_NAME
        }
        joblib.dump(model_data, MODEL_PATH)
        print("Done.")