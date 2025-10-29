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
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# --- Configuration ---
GRID_W = 100  # 800 / 8
GRID_H = 75   # 600 / 8
TARGET_ASPECT_RATIO = 4.0 / 3.0
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
MODEL_PATH = "wildlife_detector_model.pkl"
DEFAULT_IMAGE_FOLDER = "images"

# --- CHOOSE YOUR ANOMALY DETECTION MODEL ---
# Options: 'iso_forest', 'one_class_svm', 'lof'
MODEL_TO_USE = 'one_class_svm' 
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

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# ... (make sure you have these imports at the top) ...

def extract_features_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts a highly enhanced set of 114 features from all 64 cells.
    """
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # --- FIX is on this line ---
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) # <-- FIX: Was COLOR_BGR_HSV
    
    features_list = []
    
    # --- Define Gabor filter parameters ---
    gabor_thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 4 orientations
    gabor_sigmas = [1.0, 3.0] # 2 scales
    
    # --- Define Haralick parameters ---
    haralick_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 4 angles
    haralick_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'] # 6 properties

    for y in range(0, gray_img.shape[0], GRID_H):
        for x in range(0, gray_img.shape[1], GRID_W):
            
            # --- Get the cell for each image type ---
            cell_gray = gray_img[y:y+GRID_H, x:x+GRID_W]
            cell_hsv = hsv_img[y:y+GRID_H, x:x+GRID_W]
            
            cell_features = []
            
            # --- Feature Set 1: Grayscale Statistics (2 features) ---
            cell_features.append(np.mean(cell_gray))
            cell_features.append(np.std(cell_gray))
            
            # --- Feature Set 2: LBP Texture (10 features) ---
            lbp = local_binary_pattern(cell_gray, LBP_POINTS, LBP_RADIUS, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), density=True)
            cell_features.extend(hist)
            
            # --- Feature Set 3: Edge Density (1 feature) ---
            edges = cv2.Canny(cell_gray, 50, 150)
            cell_features.append(np.sum(edges > 0) / cell_gray.size)

            # --- Feature Set 4: Color Histograms (EXPANDED: 48 features) ---
            # Replaced simple mean/std with full histograms
            h, s, v = cv2.split(cell_hsv)
            h_hist, _ = np.histogram(h.ravel(), bins=16, range=[0, 180], density=True)
            s_hist, _ = np.histogram(s.ravel(), bins=16, range=[0, 256], density=True)
            v_hist, _ = np.histogram(v.ravel(), bins=16, range=[0, 256], density=True)
            cell_features.extend(h_hist) # 16 features
            cell_features.extend(s_hist) # 16 features
            cell_features.extend(v_hist) # 16 features

            # --- Feature Set 5: Haralick Texture (EXPANDED: 24 features) ---
            # Use 4 angles and 6 properties
            glcm = graycomatrix(cell_gray, distances=[1], angles=haralick_angles, levels=256, symmetric=True, normed=True)
            for prop in haralick_props:
                # ravel() flattens the (1, 4) array from 4 angles
                cell_features.extend(graycoprops(glcm, prop).ravel()) 
            # 6 properties * 4 angles = 24 features

            # --- Feature Set 6: Gabor Filters (NEW: 16 features) ---
            # Powerful texture filters at different scales/orientations
            gabor_responses = []
            for theta in gabor_thetas:
                for sigma in gabor_sigmas:
                    kernel = cv2.getGaborKernel((5, 5), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(cell_gray, cv2.CV_32F, kernel)
                    gabor_responses.append(np.mean(filtered)) # Mean of response
                    gabor_responses.append(np.std(filtered))  # Std dev of response
            cell_features.extend(gabor_responses)
            # 4 thetas * 2 sigmas * 2 stats = 16 features

            # --- Feature Set 7: Sobel Edge Statistics (NEW: 6 features) ---
            # Mean/Std of 1st and 2nd derivatives (edge magnitude/direction)
            sobelx = cv2.Sobel(cell_gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(cell_gray, cv2.CV_64F, 0, 1, ksize=5)
            magnitude = cv2.magnitude(sobelx, sobely)
            cell_features.extend([np.mean(sobelx), np.std(sobelx),
                                  np.mean(sobely), np.std(sobely),
                                  np.mean(magnitude), np.std(magnitude)])

            # --- Feature Set 8: Hu Moments (NEW: 7 features) ---
            # 7 scale/rotation/translation-invariant shape descriptors
            moments = cv2.moments(cell_gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log-scale to make values more comparable
            for i in range(7):
                hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(np.abs(hu_moments[i]) + 1e-7)
            cell_features.extend(hu_moments)
            
            # --- Total: 2+10+1+48+24+16+6+7 = 114 features ---
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
            # Your extract_features_from_image function is called here
            features = extract_features_from_image(processed_img) 
            all_features.append(features)

    if not all_features:
        print("❌ Error: No valid images found for training after processing.")
        return

    X_train = np.vstack(all_features)
    print(f"✅ Successfully extracted {X_train.shape[0]} feature vectors (original dim: {X_train.shape[1]}).")
    
    # 1. Create and fit the scaler
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # --- NEW STEP: FIT AND APPLY PCA ---
    print("Fitting PCA (n_components=3)...")
    pca_model = PCA(n_components=3)
    # Fit and transform the scaled data
    X_train_pca = pca_model.fit_transform(X_train_scaled)
    print(f"✅ Features reduced to {X_train_pca.shape[1]} components.")
    # ------------------------------------
    
    # 2. Create and fit the chosen anomaly detection model
    print(f"--- 💡 Training Model: {MODEL_TO_USE} ---")
    
    model = None
    contamination_level = 0.01 

    if MODEL_TO_USE == 'iso_forest':
        model = IsolationForest(contamination=contamination_level, random_state=42, n_jobs=-1)
    elif MODEL_TO_USE == 'one_class_svm':
        model = OneClassSVM(nu=contamination_level, kernel="rbf", gamma="auto")
    elif MODEL_TO_USE == 'lof':
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination_level, novelty=True)
    else:
        print(f"❌ Error: Unknown model '{MODEL_TO_USE}'.")
        return

    # Train on the new PCA-transformed data
    model.fit(X_train_pca) 
    
    # 3. Save ALL models to a single pickle file
    models = {
        'scaler': scaler, 
        'pca_model': pca_model,  # <-- SAVE THE PCA MODEL
        'model': model,
        'model_name': MODEL_TO_USE
    }
    joblib.dump(models, MODEL_PATH)
    
    print("\n--- ✅ Training Complete ---")
    print(f"Models (Scaler, PCA, and {MODEL_TO_USE}) saved to '{MODEL_PATH}'")

def run_prediction(args):
    """
    Executes the model prediction pipeline on a single image.
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
    pca_model = models['pca_model'] # <-- LOAD THE PCA MODEL
    model = models['model']
    model_name = models.get('model_name', 'unknown')
    print(f"Successfully loaded model: {model_name} (with Scaler and PCA)")

    # 4. Process the image
    processed_img = process_image(image_path)
    if processed_img is None:
        print("Image was discarded: did not meet 800x600 or 4:3 requirements.")
        return

    # 5. Extract features, scale, and predict
    # Your extract_features_from_image function is called here
    features = extract_features_from_image(processed_img) 
    features_scaled = scaler.transform(features)
    
    # --- NEW STEP: APPLY PCA TRANSFORM ---
    features_pca = pca_model.transform(features_scaled)
    print(f"Features transformed: {features.shape[1]} -> {features_pca.shape[1]} components")
    # -------------------------------------
    
    # Predict on the PCA-transformed features
    preds = model.predict(features_pca) 

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