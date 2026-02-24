# config.py
"""
Configuration file for Face Mask Detection System
All project settings are centralized here
"""

import os
import logging
from typing import Dict, Tuple, Any

# ============================================
# SET YOUR DATASET PATH HERE
# ============================================
CUSTOM_DATASET_PATH = r'd:\Face_Mask_detection\dataset'

# ============================================
# Project Paths
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path resolution
if CUSTOM_DATASET_PATH and os.path.exists(CUSTOM_DATASET_PATH):
    DATASET_PATH = CUSTOM_DATASET_PATH
else:
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset')

# Other paths
MODELS_PATH = os.path.join(BASE_DIR, 'models')
OUTPUT_PATH = os.path.join(BASE_DIR, 'output')
SCREENSHOTS_PATH = os.path.join(BASE_DIR, 'screenshots')
CASCADES_PATH = os.path.join(BASE_DIR, 'cascades')
LOGS_PATH = os.path.join(OUTPUT_PATH, 'logs')

# Create directories
for path in [MODELS_PATH, OUTPUT_PATH, SCREENSHOTS_PATH, CASCADES_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)

# ============================================
# Dataset Configuration
# ============================================
CATEGORIES = ['with_mask', 'without_mask']
NUM_CLASSES = len(CATEGORIES)

# ============================================
# Model Configuration
# ============================================
# MobileNetV2 specific parameters
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32

# Training parameters
INITIAL_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.0001
EPOCHS = 20
FINE_TUNE_EPOCHS = 10

# Model paths
BEST_MODEL_PATH = os.path.join(MODELS_PATH, 'best_mobilenetv2.h5')
FINAL_MODEL_PATH = os.path.join(MODELS_PATH, 'final_mobilenetv2.h5')
MODEL_ARCHITECTURE_PATH = os.path.join(OUTPUT_PATH, 'model_architecture.png')

# ============================================
# Webcam Configuration
# ============================================
WEBCAM_ID = 0  # 0 = default camera, 1 = external camera
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
WEBCAM_FPS = 30

# ============================================
# Face Detection Parameters
# ============================================
FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_SIZE = (60, 60)
MAX_FACE_SIZE = (300, 300)
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5

# ============================================
# Tracking Parameters
# ============================================
TRACKING_TIMEOUT = 3.0  # Seconds before forgetting a face
MAX_TRACKING_HISTORY = 30  # Max history entries per face
TRACKING_IOU_THRESHOLD = 0.3  # IoU threshold for same person

# ============================================
# Mask Classification Parameters
# ============================================
MASK_DETECTION_THRESHOLD = 0.25  # Minimum pixel percentage for mask detection
SKIN_DETECTION_THRESHOLD = 0.15  # Maximum skin percentage for covered nose
MASK_CLASSIFICATION_HISTORY = 5  # Frames for temporal smoothing

# Mask Types for Classification
MASK_TYPES: Dict[str, Dict[str, Any]] = {
    'surgical': {'color': (255, 0, 0), 'name': 'Surgical', 'score': 100},     # Blue
    'n95': {'color': (0, 255, 0), 'name': 'N95', 'score': 100},               # Green
    'cloth': {'color': (0, 165, 255), 'name': 'Cloth', 'score': 90},          # Orange
    'black': {'color': (100, 100, 100), 'name': 'Black', 'score': 90},        # Gray
    'white': {'color': (255, 255, 255), 'name': 'White', 'score': 95},        # White
    'proper': {'color': (0, 255, 0), 'name': 'Proper', 'score': 100},         # Green
    'improper': {'color': (0, 0, 255), 'name': 'Improper', 'score': 30},      # Red
    'below_nose': {'color': (0, 165, 255), 'name': 'Below Nose', 'score': 60}, # Orange
    'no_mask': {'color': (0, 0, 255), 'name': 'No Mask', 'score': 0},         # Red
    'unknown': {'color': (128, 128, 128), 'name': 'Unknown', 'score': 50}     # Gray
}

# Compliance Thresholds
COMPLIANCE_THRESHOLDS: Dict[str, int] = {
    'excellent': 95,  # >95% = Excellent (Green)
    'good': 80,       # 80-95% = Good (Yellow)
    'poor': 0         # <80% = Poor (Red)
}

# ============================================
# Colors - Separate for OpenCV and Matplotlib
# ============================================

# For OpenCV (BGR format - 0-255 range)
COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    'with_mask': (0, 255, 0),          # Green
    'without_mask': (0, 0, 255),        # Red
    'proper': (0, 255, 0),              # Green
    'improper': (0, 0, 255),            # Red
    'below_nose': (0, 165, 255),        # Orange
    'surgical': (255, 0, 0),            # Blue
    'n95': (0, 255, 0),                 # Green
    'cloth': (0, 165, 255),             # Orange
    'black': (100, 100, 100),           # Gray
    'white': (255, 255, 255),           # White
    'text': (255, 255, 255),            # White
    'background': (0, 0, 0),            # Black
    'success': (0, 255, 0),             # Green
    'warning': (0, 255, 255),           # Yellow
    'error': (0, 0, 255),               # Red
    'info': (255, 255, 0),              # Cyan
}

# For Matplotlib (RGB format - 0-1 range)
COLORS_RGB: Dict[str, Tuple[float, float, float]] = {
    'with_mask': (0, 1, 0),             # Green
    'without_mask': (1, 0, 0),           # Red
    'proper': (0, 1, 0),                 # Green
    'improper': (1, 0, 0),               # Red
    'below_nose': (0, 0.65, 1),          # Orange
    'surgical': (0, 0, 1),               # Blue
    'n95': (0, 1, 0),                    # Green
    'cloth': (0, 0.65, 1),               # Orange
    'black': (0.4, 0.4, 0.4),            # Gray
    'white': (1, 1, 1),                  # White
    'text': (1, 1, 1),                   # White
    'background': (0, 0, 0),             # Black
    'success': (0, 1, 0),                # Green
    'warning': (1, 1, 0),                # Yellow
    'error': (1, 0, 0),                  # Red
    'info': (0, 1, 1),                   # Cyan
}

# For backward compatibility
COLORS = COLORS_BGR

# ============================================
# Logging Configuration
# ============================================
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOGS_PATH, 'app.log')

# ============================================
# Validation Functions
# ============================================
def validate_paths() -> bool:
    """
    Validate that all required paths are accessible
    
    Returns:
        bool: True if all paths are valid
    """
    paths_to_check = [
        ('Dataset', DATASET_PATH),
        ('Models', MODELS_PATH),
        ('Output', OUTPUT_PATH),
        ('Screenshots', SCREENSHOTS_PATH),
        ('Cascades', CASCADES_PATH),
        ('Logs', LOGS_PATH)
    ]
    
    all_valid = True
    for name, path in paths_to_check:
        if os.path.exists(path):
            if not os.access(path, os.W_OK):
                logging.warning(f"{name} path not writable: {path}")
                all_valid = False
        else:
            try:
                os.makedirs(path, exist_ok=True)
                logging.info(f"Created {name} path: {path}")
            except Exception as e:
                logging.error(f"Cannot create {name} path {path}: {e}")
                all_valid = False
    
    return all_valid

def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of current configuration
    
    Returns:
        dict: Configuration summary
    """
    return {
        'dataset_path': DATASET_PATH,
        'models_path': MODELS_PATH,
        'output_path': OUTPUT_PATH,
        'image_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'categories': CATEGORIES,
        'num_classes': NUM_CLASSES,
        'initial_lr': INITIAL_LEARNING_RATE,
        'fine_tune_lr': FINE_TUNE_LEARNING_RATE,
        'epochs': EPOCHS,
        'fine_tune_epochs': FINE_TUNE_EPOCHS,
        'webcam_resolution': f"{WEBCAM_WIDTH}x{WEBCAM_HEIGHT}",
        'min_face_size': MIN_FACE_SIZE,
        'tracking_timeout': TRACKING_TIMEOUT,
    }

# ============================================
# Print configuration summary (only when run directly)
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FACE MASK DETECTION - CONFIGURATION")
    print("="*60)
    
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\nPaths:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Models: {MODELS_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Screenshots: {SCREENSHOTS_PATH}")
    print(f"  Cascades: {CASCADES_PATH}")
    print(f"  Logs: {LOGS_PATH}")
    
    print("\n" + "="*60)