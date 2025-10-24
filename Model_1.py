#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wildlife Anomaly Detector
=========================

This single script handles both training and prediction for a 
grid-based anomaly detection model.

Usage:
  Train the model:
    python this_script.py train

  Run prediction:
    python this_script.py predict
"""

import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
import joblib
import random
import argparse
import sys

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from skimage.feature import graycomatrix, graycoprops

# --- Configuration ---
GRID_W = 100  # 800 / 8
GRID_H = 75   # 600 / 8
TARGET_ASPECT_RATIO = 4.0 / 3.0
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
MODEL_PATH = "wildlife_detector_model.pkl"
DEFAULT_IMAGE_FOLDER = "images"

# --- 💡 CHOOSE YOUR ANOMALY DETECTION MODEL ---
# Options: 'iso_forest', 'one_class_svm', 'lof'
MODEL_TO_USE = 'lof' 
# ==========================================


# --- Common Helper Functions (Unchanged) ---

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
    if w > 800 or h > 600:
        img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)

    h, w, _ = img.shape
    if w < 800 or h < 600:
        return None
        
    return img

def extract_features_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts an enhanced set of features from all 64 cells in a single BGR image.
    """
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    features_list = []
    
    for y in range(0, gray_img.shape[0], GRID_H):
        for x in range(0, gray_img.shape[1], GRID_W):
            
            # --- Get the cell for each image type ---
            cell_gray = gray_img[y:y+GRID_H, x:x+GRID_W]
            cell_hsv = hsv_img[y:y+GRID_H, x:x+GRID_W]
            
            cell_features = []
            
            # --- Feature Set 1: Grayscale Statistics (Original) ---
            cell_features.append(np.mean(cell_gray))
            cell_features.append(np.std(cell_gray))
            
            # --- Feature Set 2: LBP Texture (Original) ---
            lbp = local_binary_pattern(cell_gray, LBP_POINTS, LBP_RADIUS, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), density=True)
            cell_features.extend(hist)
            
            # --- Feature Set 3: Edge Density (Original) ---
            edges = cv2.Canny(cell_gray, 50, 150)
            cell_features.append(np.sum(edges > 0) / cell_gray.size)

            # --- Feature Set 4: Color Statistics (NEW) ---
            # Split HSV channels and get mean/std for each
            h, s, v = cv2.split(cell_hsv)
            cell_features.append(np.mean(h))
            cell_features.append(np.std(h))
            cell_features.append(np.mean(s))
            cell_features.append(np.std(s))
            cell_features.append(np.mean(v))
            cell_features.append(np.std(v))

            # --- Feature Set 5: Haralick Texture (NEW) ---
            # Compute Gray-Level Co-occurrence Matrix (GLCM)
            # Requires image to be uint8
            glcm = graycomatrix(cell_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            
            # Extract properties from GLCM
            cell_features.append(graycoprops(glcm, 'contrast')[0, 0])
            cell_features.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            cell_features.append(graycoprops(glcm, 'homogeneity')[0, 0])
            cell_features.append(graycoprops(glcm, 'energy')[0, 0])
            
            features_list.append(np.array(cell_features))
            
    return np.vstack(features_list)

def draw_grid_visualization(img: np.ndarray, grid_map: np.ndarray) -> np.ndarray:
    """
    Draws the visualization grid on the image.
    """
    def apply_dither_effect(cell: np.ndarray) -> np.ndarray:
        h, w, _ = cell.shape
        overlay = np.zeros_like(cell, dtype=np.uint8)
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                cv2.circle(overlay, (x, y), 1, (200, 200, 200), -1)
        return cv2.addWeighted(cell, 0.5, overlay, 0.5, 0)

    cell_number = 1
    for i in range(8): # 8 rows
        for j in range(8): # 8 cols
            y1, x1 = i * GRID_H, j * GRID_W
            y2, x2 = y1 + GRID_H, x1 + GRID_W
            
            if grid_map[i, j] == 1:
                img[y1:y2, x1:x2] = apply_dither_effect(img[y1:y2, x1:x2])
                
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(img, str(cell_number), (x1 + 3, y1 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cell_number += 1
    return img

# --- Main Functions (Train vs. Predict) ---

def run_training(args):
    """
    Executes the model training pipeline.
    """
    print("--- 🚀 Starting Model Training ---")
    training_folder = args.folder
    
    os.makedirs(training_folder, exist_ok=True)
    image_paths = [os.path.join(training_folder, f) for f in os.listdir(training_folder) 
                   if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    if not image_paths:
        print(f"❌ Error: No images found in '{training_folder}'.")
        return

    all_features = []
    print(f"Processing {len(image_paths)} images from '{training_folder}'...")
    for path in image_paths:
        processed_img = process_image(path)
        if processed_img is not None:
            features = extract_features_from_image(processed_img)
            all_features.append(features)

    if not all_features:
        print("❌ Error: No valid images found for training after processing.")
        return

    X_train = np.vstack(all_features)
    print(f"✅ Successfully extracted {X_train.shape[0]} feature vectors for training.")
    
    # 1. Create and fit the scaler
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # === MODIFIED SECTION ===
    # 2. Create and fit the chosen anomaly detection model
    print(f"--- 💡 Training Model: {MODEL_TO_USE} ---")
    
    model = None
    # Set a low contamination if you trust your training data is "clean"
    contamination_level = 0.01 

    if MODEL_TO_USE == 'iso_forest':
        # Good all-rounder, fast.
        model = IsolationForest(contamination=contamination_level, random_state=42, n_jobs=-1)
    
    elif MODEL_TO_USE == 'one_class_svm':
        # Good for high-dimensional data, can be slow to train.
        # 'nu' is similar to contamination.
        model = OneClassSVM(nu=contamination_level, kernel="rbf", gamma="auto")
        
    elif MODEL_TO_USE == 'lof':
        # Good at finding local anomalies, but must set novelty=True.
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination_level, novelty=True)

    else:
        print(f"❌ Error: Unknown model '{MODEL_TO_USE}'.")
        print("Please set MODEL_TO_USE to 'iso_forest', 'one_class_svm', or 'lof' at the top.")
        return

    model.fit(X_train_scaled)
    # ==========================
    
    # 3. Save both models to a single pickle file
    models = {
        'scaler': scaler, 
        'model': model, # Save the generic 'model' variable
        'model_name': MODEL_TO_USE # Save the name
    }
    joblib.dump(models, MODEL_PATH)
    
    print("\n--- ✅ Training Complete ---")
    print(f"Models saved to '{MODEL_PATH}'")

def run_prediction(args):
    """
    Executes the model prediction pipeline on a single image.
    (This function is unchanged)
    """
    print("--- 🔍 Running Model Prediction ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at '{MODEL_PATH}'.")
        print(f"Please run: python {sys.argv[0]} train")
        return

    image_path = args.image
    if image_path:
        if not os.path.exists(image_path):
            print(f"❌ Error: Specified image file not found: {image_path}")
            return
        print(f"--- Testing specified image: {image_path} ---")
    else:
        if not os.path.isdir(DEFAULT_IMAGE_FOLDER):
            print(f"❌ Error: Default folder '{DEFAULT_IMAGE_FOLDER}' not found.")
            return
        all_images = [f for f in os.listdir(DEFAULT_IMAGE_FOLDER) 
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        if not all_images:
            print(f"❌ No images found in '{DEFAULT_IMAGE_FOLDER}'.")
            return
        random_image_name = random.choice(all_images)
        image_path = os.path.join(DEFAULT_IMAGE_FOLDER, random_image_name)
        print(f"--- Testing random image: {random_image_name} ---")

    # 3. Load models
    models = joblib.load(MODEL_PATH)
    scaler = models['scaler']
    model = models['model']
    model_name = models.get('model_name', 'unknown')
    print(f"Successfully loaded model: {model_name}")

    # 4. Process the image
    processed_img = process_image(image_path)
    if processed_img is None:
        print("Image was discarded: did not meet 800x600 or 4:3 aspect ratio requirements.")
        return

    # 5. Extract features, scale, and predict
    features = extract_features_from_image(processed_img)
    features_scaled = scaler.transform(features)
    preds = model.predict(features_scaled)

    # 6. Visualize and show the result
    grid_map = (preds == -1).astype(int).reshape((8, 8))
    final_image = draw_grid_visualization(processed_img.copy(), grid_map)

    output_filename = f"test_output_{os.path.basename(image_path)}"
    cv2.imwrite(output_filename, final_image)
    print(f"✅ Output saved to '{output_filename}'")
    
    window_title = f"Test Result: {os.path.basename(image_path)} (Model: {model_name})"
    cv2.imshow(window_title, final_image)
    print("Press any key to close the image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    Main entry point with command-line argument parsing.
    (This function is unchanged)
    """
    parser = argparse.ArgumentParser(
        description="Wildlife Anomaly Detector - Train and Predict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", required=True, 
                                       help="Available commands")

    # --- Train Sub-command ---
    train_parser = subparsers.add_parser("train", help="Train the anomaly detection model")
    train_parser.add_argument(
        "-f", "--folder", type=str, default=DEFAULT_IMAGE_FOLDER,
        help=f"Folder containing training images (default: {DEFAULT_IMAGE_FOLDER})"
    )
    train_parser.set_defaults(func=run_training)

    # --- Predict Sub-command ---
    predict_parser = subparsers.add_parser("predict", 
                                           help="Run prediction on an image")
    predict_parser.add_argument(
        "-i", "--image", type=str,
        help="Path to a specific image to test. (default: random image from 'images' folder)"
    )
    predict_parser.set_defaults(func=run_prediction)
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()