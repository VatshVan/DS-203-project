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

  Train the model using a specific folder:
    python this_script.py train -f /path/to/training_images
  
  Run prediction on a random image from the default folder:
    python this_script.py predict

  Run prediction on a specific image:
    python this_script.py predict -i /path/to/my_image.jpg
"""

import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import random
import argparse
import sys

# --- Configuration ---
# An 8x8 grid on an 800x600 image.
GRID_W = 100  # 800 / 8
GRID_H = 75   # 600 / 8
TARGET_ASPECT_RATIO = 4.0 / 3.0

# LBP parameters
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

# File paths
MODEL_PATH = "wildlife_detector_model.pkl"
DEFAULT_IMAGE_FOLDER = "images"

# --- Common Helper Functions ---

def process_image(path: str) -> np.ndarray | None:
    """
    Loads an image, enforces 4:3 aspect ratio by center-cropping,
    and resizes to exactly 800x600.
    
    Returns the BGR image or None if the image is invalid or too small.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read image at {path}")
        return None
    
    h, w, _ = img.shape
    current_aspect_ratio = w / h
    
    # 1. Enforce 4:3 aspect ratio by center-cropping
    if not np.isclose(current_aspect_ratio, TARGET_ASPECT_RATIO):
        if current_aspect_ratio > TARGET_ASPECT_RATIO: # Image is wider
            new_w = int(TARGET_ASPECT_RATIO * h)
            x_start = (w - new_w) // 2
            img = img[:, x_start:x_start + new_w]
        else: # Image is taller
            new_h = int(w / TARGET_ASPECT_RATIO)
            y_start = (h - new_h) // 2
            img = img[y_start:y_start + new_h, :]
    
    # 2. Scale down if larger than 800x600
    h, w, _ = img.shape
    if w > 800 or h > 600:
        img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)

    # 3. Discard if smaller than 800x600
    h, w, _ = img.shape
    if w < 800 or h < 600:
        # print(f"Info: Discarding small image: {path} ({w}x{h})")
        return None
        
    return img

def extract_features_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts features from all 64 cells in a single BGR image.
    Converts to grayscale internally for feature extraction.
    """
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    features_list = []
    
    for y in range(0, gray_img.shape[0], GRID_H):
        for x in range(0, gray_img.shape[1], GRID_W):
            cell = gray_img[y:y+GRID_H, x:x+GRID_W]
            
            # --- Feature Extraction Logic ---
            cell_features = []
            # 1. Mean intensity
            cell_features.append(np.mean(cell))
            # 2. Standard deviation
            cell_features.append(np.std(cell))
            
            # 3. LBP Texture Histogram
            lbp = local_binary_pattern(cell, LBP_POINTS, LBP_RADIUS, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), density=True)
            cell_features.extend(hist)
            
            # 4. Edge density
            edges = cv2.Canny(cell, 50, 150)
            cell_features.append(np.sum(edges > 0) / cell.size)
            
            features_list.append(np.array(cell_features))
            
    return np.vstack(features_list)

def draw_grid_visualization(img: np.ndarray, grid_map: np.ndarray) -> np.ndarray:
    """
    Draws the visualization grid on the image.
    Highlights anomalous cells (where grid_map == 1).
    """
    def apply_dither_effect(cell: np.ndarray) -> np.ndarray:
        """Applies a semi-transparent overlay to anomalous cells."""
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
            
            # Apply visual effect if anomaly (1)
            if grid_map[i, j] == 1:
                img[y1:y2, x1:x2] = apply_dither_effect(img[y1:y2, x1:x2])
                
            # Draw grid rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            # Draw cell number
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
        print("Please add 'normal' (no wildlife) images to this folder to train the model.")
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
        print("Please ensure your images are at least 800x600.")
        return

    X_train = np.vstack(all_features)
    print(f"✅ Successfully extracted {X_train.shape[0]} feature vectors for training.")
    
    # 1. Create and fit the scaler
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 2. Create and fit the Isolation Forest model
    print("Training IsolationForest...")
    # Contamination: The expected proportion of anomalies in the training data.
    # Set this to a low value if you assume your training data is "clean".
    iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_scaled)
    
    # 3. Save both models to a single pickle file
    models = {
        'scaler': scaler, 
        'model': iso_forest,
        'model_name': 'iso_forest'
    }
    joblib.dump(models, MODEL_PATH)
    
    print("\n--- ✅ Training Complete ---")
    print(f"Models saved to '{MODEL_PATH}'")

def run_prediction(args):
    """
    Executes the model prediction pipeline on a single image.
    """
    print("--- 🔍 Running Model Prediction ---")
    
    # 1. Check for model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at '{MODEL_PATH}'.")
        print("Please run the 'train' command first:")
        print(f"  python {sys.argv[0]} train")
        return

    # 2. Find image to predict
    image_path = args.image
    if image_path:
        if not os.path.exists(image_path):
            print(f"❌ Error: Specified image file not found: {image_path}")
            return
        print(f"--- Testing specified image: {image_path} ---")
    else:
        # No specific image given, pick a random one
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
    # Anomalies are -1. Map to 1 for anomaly, 0 for normal.
    grid_map = (preds == -1).astype(int).reshape((8, 8))
    
    # Draw on a copy of the BGR image
    final_image = draw_grid_visualization(processed_img.copy(), grid_map)

    # Save the output for review
    output_filename = f"test_output_{os.path.basename(image_path)}"
    cv2.imwrite(output_filename, final_image)
    print(f"✅ Output saved to '{output_filename}'")
    
    # Display in a window
    window_title = f"Test Result: {os.path.basename(image_path)} (Model: {model_name})"
    cv2.imshow(window_title, final_image)
    print("Press any key to close the image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    Main entry point with command-line argument parsing.
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
    
    # Show help if no command is given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()