import cv2
import numpy as np
import glob
import os
import sys

# --- 1. Define Constants ---
TARGET_W, TARGET_H = 800, 600
TARGET_ASPECT_RATIO = TARGET_W / TARGET_H  # 4.0 / 3.0
GRID_ROWS, GRID_COLS = 8, 8
GRID_W = TARGET_W // GRID_COLS  # 100 pixels wide
GRID_H = TARGET_H // GRID_ROWS  # 75 pixels high

IMAGE_DIR = "images"
LABEL_DIR = "labels"
WINDOW_NAME = "Wildlife Labeler"
ACTIVE_LEARNING_LIST = "to_label.txt" # The file to check for

# --- 2. Your Helper Functions ---
# (process_image and draw_grid_visualization are unchanged)

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
    for i in range(GRID_ROWS): # 8 rows
        for j in range(GRID_COLS): # 8 cols
            y1, x1 = i * GRID_H, j * GRID_W
            y2, x2 = y1 + GRID_H, x1 + GRID_W
            
            if grid_map[i, j] == 1:
                img[y1:y2, x1:x2] = apply_dither_effect(img[y1:y2, x1:x2])
                
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(img, str(cell_number), (x1 + 3, y1 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cell_number += 1
    return img

# --- 3. Interactive Labeling Logic ---

current_grid_map = None
current_image_clean = None
current_image_display = None

def on_mouse_click(event, x, y, flags, param):
    """
    This function is called by OpenCV whenever the mouse is clicked.
    """
    global current_grid_map, current_image_clean, current_image_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        i = y // GRID_H
        j = x // GRID_W
        
        if 0 <= i < GRID_ROWS and 0 <= j < GRID_COLS:
            current_grid_map[i, j] = 1 - current_grid_map[i, j]
            
            temp_img = current_image_clean.copy()
            current_image_display = draw_grid_visualization(temp_img, current_grid_map)
            cv2.imshow(WINDOW_NAME, current_image_display)

# --- 4. TWEAKED MAIN FUNCTION ---

def main():
    """
    The main application loop.
    Checks for 'to_label.txt' to decide which images to load.
    """
    global current_grid_map, current_image_clean, current_image_display

    print("Starting Wildlife Labeler...")
    print("---------------------------------")
    print("INSTRUCTIONS:")
    print("  - Click a cell to toggle its label (0 or 1).")
    print("  - 's' : Save the current label and move to the next image.")
    print("  - 'n' : Skip this image (no label saved).")
    print("  - 'q' : Quit the application.")
    print("---------------------------------")

    os.makedirs(LABEL_DIR, exist_ok=True)
    
    image_paths = []
    is_active_learning_mode = os.path.exists(ACTIVE_LEARNING_LIST)

    # --- THIS IS THE NEW LOGIC ---
    if is_active_learning_mode:
        print(f"--- Found '{ACTIVE_LEARNING_LIST}'. Starting Active Learning session. ---")
        try:
            with open(ACTIVE_LEARNING_LIST, 'r') as f:
                # Read paths, strip newline characters, ignore empty lines
                image_paths = [line.strip() for line in f if line.strip()]
            
            if not image_paths:
                print(f"Warning: '{ACTIVE_LEARNING_LIST}' is empty. Nothing to label.")
                return
        except IOError as e:
            print(f"Error: Could not read file {ACTIVE_LEARNING_LIST}. {e}")
            return
    else:
        print("--- No active learning list found. Starting 'Seed Mode'. ---")
        print("--- Scanning all images and skipping previously labeled ones. ---")
        image_types = ("*.jpg", "*.jpeg", "*.png")
        for img_type in image_types:
            image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, img_type)))
        
        if not image_paths:
            print(f"Error: No images found in '{IMAGE_DIR}'. Please add images and try again.")
            return

    # --- End of new logic ---

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

    images_processed_in_batch = 0
    total_images_in_batch = len(image_paths)

    for img_path in image_paths:
        # Check if the path from the file is valid
        if not os.path.exists(img_path):
            print(f"Warning: Image path not found, skipping: {img_path}")
            continue

        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, f"{img_basename}.npy")

        # In Seed Mode, we skip already labeled files.
        # In Active Learning Mode, we DON'T skip, because find_confusing.py
        # already confirmed it needs labeling.
        if not is_active_learning_mode:
            if os.path.exists(label_path):
                print(f"Skipping (already labeled): {img_basename}")
                continue
        
        img_clean = process_image(img_path)
        if img_clean is None:
            continue
        
        print(f"\nNow labeling: {img_basename}")

        current_image_clean = img_clean
        
        # Check if a label file exists (e.g., from a partial session)
        if os.path.exists(label_path):
            print("... loading existing partial label.")
            current_grid_map = np.load(label_path)
        else:
            current_grid_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
            
        current_image_display = draw_grid_visualization(current_image_clean.copy(), current_grid_map)

        while True:
            cv2.imshow(WINDOW_NAME, current_image_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                np.save(label_path, current_grid_map)
                print(f"-> Saved: {label_path}")
                images_processed_in_batch += 1
                break 
            
            if key == ord('n'):
                print(f"-> Skipped: {img_basename}")
                images_processed_in_batch += 1
                break 
            
            if key == ord('q'):
                print("Quitting early...")
                cv2.destroyAllWindows()
                
                if is_active_learning_mode:
                    print(f"--- WARNING: Batch not finished. '{ACTIVE_LEARNING_LIST}' was NOT deleted. ---")
                    print("--- Run labeler again to finish the batch. ---")
                
                return 

    # --- After the loop finishes ---
    if is_active_learning_mode:
        print(f"\n--- Active learning batch complete ({images_processed_in_batch} / {total_images_in_batch}) ---")
        
        # Only delete the list if the user *finished* the whole batch
        # This prevents quitting halfway and losing your list
        if images_processed_in_batch == total_images_in_batch:
            try:
                os.remove(ACTIVE_LEARNING_LIST)
                print(f"--- Successfully processed batch. Deleting '{ACTIVE_LEARNING_LIST}'. ---")
            except OSError as e:
                print(f"Error: Could not delete '{ACTIVE_LEARNING_LIST}'. {e}")
        else:
             print(f"--- WARNING: Batch not finished. '{ACTIVE_LEARNING_LIST}' was NOT deleted. ---")
             print("--- Run labeler again to finish the batch. ---")
    else:
        print("\nAll 'Seed Mode' images processed. Exiting.")
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()