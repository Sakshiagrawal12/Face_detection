# utils/mask_classifier.py
"""
Mask Classification Module
Classifies mask types and checks proper wearing using computer vision
"""

import cv2
import numpy as np
import os
import logging
from typing import Tuple, List, Dict, Any, Optional
from collections import deque
from config import COLORS_BGR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MaskClassifier:
    """
    FEATURE 4: Mask Type Classification
    Classifies different types of masks and checks proper wearing
    
    Attributes:
        face_cascade: Haar cascade for face detection
        nose_cascade: Haar cascade for nose detection (optional)
        mask_ranges: HSV color ranges for different mask types
        history: Recent classifications for temporal smoothing
    """
    
    def __init__(self, history_size: int = 5):
        """
        Initialize mask classifier
        
        Args:
            history_size: Number of frames to keep for temporal smoothing
        """
        self.history_size = history_size
        self.history = deque(maxlen=history_size)
        
        # Load cascades with fallback options
        self.face_cascade = self._load_cascade('haarcascade_frontalface_default.xml')
        self.nose_cascade = self._load_cascade(['haarcascade_nose.xml', 'haarcascade_mcs_nose.xml'])
        
        # Extended color ranges for mask types (in HSV)
        self.mask_ranges = {
            'surgical': {
                'lower': np.array([90, 40, 40]),
                'upper': np.array([130, 255, 255]),
                'color': COLORS_BGR.get('surgical', (255, 0, 0))
            },
            'n95': {
                'lower': np.array([35, 30, 30]),
                'upper': np.array([85, 255, 255]),
                'color': COLORS_BGR.get('n95', (0, 255, 0))
            },
            'cloth': {
                'lower': np.array([0, 30, 30]),
                'upper': np.array([20, 255, 255]),
                'color': COLORS_BGR.get('cloth', (0, 165, 255))
            },
            'black': {  # Common for cloth masks
                'lower': np.array([0, 0, 0]),
                'upper': np.array([180, 255, 50]),
                'color': (100, 100, 100)
            },
            'white': {  # Common for surgical masks
                'lower': np.array([0, 0, 200]),
                'upper': np.array([180, 30, 255]),
                'color': (255, 255, 255)
            }
        }
        
        # Skin tone range for detecting uncovered nose
        self.skin_ranges = [
            (np.array([0, 20, 70]), np.array([20, 150, 255])),  # Light skin
            (np.array([170, 20, 70]), np.array([180, 150, 255])),  # Darker skin
        ]
        
        # Confidence thresholds
        self.mask_threshold = 0.25  # Minimum pixel percentage for mask detection
        self.skin_threshold = 0.15  # Maximum skin percentage for covered nose
        
        logging.info(f"MaskClassifier initialized with {len(self.mask_ranges)} mask types")
    
    def _load_cascade(self, cascade_names: List[str]) -> Optional[cv2.CascadeClassifier]:
        """
        Load Haar cascade with multiple fallback options
        
        Args:
            cascade_names: List of possible cascade filenames
            
        Returns:
            CascadeClassifier or None if not found
        """
        if isinstance(cascade_names, str):
            cascade_names = [cascade_names]
        
        # Search paths
        search_paths = [
            cv2.data.haarcascades,
            'cascades/',
            './',
            os.path.join(os.path.dirname(__file__), '..', 'cascades')
        ]
        
        for cascade_name in cascade_names:
            for base_path in search_paths:
                cascade_path = os.path.join(base_path, cascade_name)
                if os.path.exists(cascade_path):
                    try:
                        cascade = cv2.CascadeClassifier(cascade_path)
                        if not cascade.empty():
                            logging.info(f"✅ Loaded cascade: {cascade_path}")
                            return cascade
                    except Exception as e:
                        logging.warning(f"Failed to load {cascade_path}: {e}")
        
        logging.warning(f"⚠️ Could not load cascade: {cascade_names}")
        return None
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division to avoid division by zero"""
        return numerator / denominator if denominator != 0 else default
    
    def _get_face_regions(self, face_roi: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract different regions of the face
        
        Args:
            face_roi: Face region image
            
        Returns:
            Dictionary of face regions
        """
        h, w = face_roi.shape[:2]
        
        # Define regions (with bounds checking)
        regions = {
            'forehead': face_roi[:h//4, :],
            'eyes': face_roi[h//4:h//2, :],
            'nose': face_roi[h//2:3*h//4, w//4:3*w//4],
            'mouth': face_roi[3*h//4:, w//4:3*w//4],
            'upper_face': face_roi[:h//2, :],
            'lower_face': face_roi[h//2:, :],
            'mask_area': face_roi[h//3:, w//4:3*w//4]  # Area where mask should be
        }
        
        # Validate regions
        for name, region in list(regions.items()):
            if region.size == 0:
                regions[name] = None
        
        return regions
    
    def check_proper_wearing(self, face_roi: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Check if mask is worn properly (covers nose)
        
        Args:
            face_roi: Face region image
            
        Returns:
            Tuple of (is_proper, list of issues)
        """
        issues = []
        
        if face_roi is None or face_roi.size == 0:
            return False, ["no_face"]
        
        try:
            h, w = face_roi.shape[:2]
            regions = self._get_face_regions(face_roi)
            nose_region = regions.get('nose')
            
            # Method 1: Nose detection using cascade (if available)
            if self.nose_cascade is not None:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                noses = self.nose_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
                )
                if len(noses) > 0:
                    issues.append("nose_detected")
            
            # Method 2: Skin detection in nose region
            if nose_region is not None and nose_region.size > 0:
                # Convert to HSV for skin detection
                hsv_nose = cv2.cvtColor(nose_region, cv2.COLOR_BGR2HSV)
                
                # Check for skin tones
                skin_pixels = 0
                total_pixels = nose_region.shape[0] * nose_region.shape[1]
                
                for lower_skin, upper_skin in self.skin_ranges:
                    skin_mask = cv2.inRange(hsv_nose, lower_skin, upper_skin)
                    skin_pixels += np.sum(skin_mask > 0)
                
                skin_percentage = self._safe_divide(skin_pixels, total_pixels)
                
                if skin_percentage > self.skin_threshold:
                    issues.append("nose_uncovered")
                    logging.debug(f"Nose skin percentage: {skin_percentage:.2f}")
            
            # Method 3: Check mask position using edge detection
            if self._is_mask_below_nose(face_roi):
                issues.append("mask_below_nose")
            
            # Method 4: Check for mask edges
            if not self._has_mask_edges(face_roi):
                issues.append("no_mask_edges_detected")
            
            is_proper = len(issues) == 0
            return is_proper, issues
            
        except Exception as e:
            logging.error(f"Error checking proper wearing: {e}")
            return False, ["error"]
    
    def _is_mask_below_nose(self, face_roi: np.ndarray) -> bool:
        """
        Check if mask is worn below the nose using edge detection
        
        Args:
            face_roi: Face region image
            
        Returns:
            True if mask appears to be below nose
        """
        try:
            h, w = face_roi.shape[:2]
            
            # Define upper and lower regions
            upper_region = face_roi[h//3:2*h//3, w//4:3*w//4]
            lower_region = face_roi[2*h//3:, w//4:3*w//4]
            
            if upper_region.size == 0 or lower_region.size == 0:
                return False
            
            # Convert to HSV
            hsv_upper = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
            
            # Detect skin in upper region
            skin_pixels_upper = 0
            for lower_skin, upper_skin in self.skin_ranges:
                skin_mask = cv2.inRange(hsv_upper, lower_skin, upper_skin)
                skin_pixels_upper += np.sum(skin_mask > 0)
            
            upper_skin_percentage = self._safe_divide(
                skin_pixels_upper, upper_region.shape[0] * upper_region.shape[1]
            )
            
            # If too much skin visible in upper region, mask is too low
            return upper_skin_percentage > self.skin_threshold * 1.5
            
        except Exception as e:
            logging.error(f"Error checking mask position: {e}")
            return False
    
    def _has_mask_edges(self, face_roi: np.ndarray) -> bool:
        """
        Detect mask edges using Canny edge detection
        
        Args:
            face_roi: Face region image
            
        Returns:
            True if mask edges are detected
        """
        try:
            h, w = face_roi.shape[:2]
            
            # Focus on lower face where mask should be
            lower_face = face_roi[h//2:, :]
            
            if lower_face.size == 0:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect edges
            edges = cv2.Canny(blurred, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size
            
            # Masks typically create strong horizontal edges
            return edge_density > 0.05
            
        except Exception as e:
            logging.error(f"Error detecting mask edges: {e}")
            return True  # Assume mask on error
    
    def classify_mask_type(self, face_roi: np.ndarray, has_mask: bool) -> Tuple[str, float, tuple]:
        """
        Classify the type of mask being worn
        
        Args:
            face_roi: Face region image
            has_mask: Whether mask was detected by base model
            
        Returns:
            Tuple of (mask_type, confidence, color)
        """
        if not has_mask:
            return 'no_mask', 1.0, COLORS_BGR.get('without_mask', (0, 0, 255))
        
        if face_roi is None or face_roi.size == 0:
            return 'unknown', 0.5, (128, 128, 128)
        
        try:
            # Check proper wearing first
            is_proper, issues = self.check_proper_wearing(face_roi)
            
            if not is_proper:
                if "nose_uncovered" in issues:
                    return 'below_nose', 0.7, COLORS_BGR.get('below_nose', (0, 165, 255))
                else:
                    return 'improper', 0.6, COLORS_BGR.get('improper', (0, 0, 255))
            
            # Focus on mask area
            regions = self._get_face_regions(face_roi)
            mask_area = regions.get('mask_area')
            
            if mask_area is None or mask_area.size == 0:
                return 'proper', 0.6, COLORS_BGR.get('proper', (0, 255, 0))
            
            # Convert to HSV for color analysis
            hsv_area = cv2.cvtColor(mask_area, cv2.COLOR_BGR2HSV)
            
            # Check each mask type
            best_match = 'proper'
            best_score = 0.3
            best_color = COLORS_BGR.get('proper', (0, 255, 0))
            
            for mask_type, config in self.mask_ranges.items():
                # Create mask for color range
                color_mask = cv2.inRange(hsv_area, config['lower'], config['upper'])
                
                # Calculate percentage
                score = self._safe_divide(np.sum(color_mask > 0), color_mask.size)
                
                if score > best_score and score > self.mask_threshold:
                    best_score = score
                    best_match = mask_type
                    best_color = config.get('color', (255, 255, 255))
            
            # Temporal smoothing
            self.history.append((best_match, best_score))
            
            # Get most common type from history
            if len(self.history) >= 3:
                types = [t for t, _ in self.history]
                from collections import Counter
                most_common = Counter(types).most_common(1)[0][0]
                if most_common != best_match:
                    logging.debug(f"Smoothed classification: {best_match} -> {most_common}")
                    best_match = most_common
            
            return best_match, best_score, best_color
            
        except Exception as e:
            logging.error(f"Error classifying mask type: {e}")
            return 'unknown', 0.5, (128, 128, 128)
    
    def get_compliance_score(self, mask_type: str, is_proper: bool) -> float:
        """
        Calculate compliance score based on mask type and proper wearing
        
        Args:
            mask_type: Type of mask detected
            is_proper: Whether mask is worn properly
            
        Returns:
            Compliance score (0-100)
        """
        base_scores = {
            'surgical': 100,
            'n95': 100,
            'cloth': 90,
            'black': 90,
            'white': 100,
            'proper': 100,
            'below_nose': 60,
            'improper': 30,
            'no_mask': 0,
            'unknown': 50
        }
        
        if not is_proper and mask_type not in ['no_mask', 'unknown']:
            return base_scores.get('below_nose', 60)
        
        return base_scores.get(mask_type, 50)
    
    def get_mask_color(self, mask_type: str) -> tuple:
        """
        Get display color for mask type
        
        Args:
            mask_type: Type of mask
            
        Returns:
            BGR color tuple
        """
        color_map = {
            'surgical': (255, 0, 0),
            'n95': (0, 255, 0),
            'cloth': (0, 165, 255),
            'black': (100, 100, 100),
            'white': (255, 255, 255),
            'proper': (0, 255, 0),
            'below_nose': (0, 165, 255),
            'improper': (0, 0, 255),
            'no_mask': (0, 0, 255),
            'unknown': (128, 128, 128)
        }
        return color_map.get(mask_type, (255, 255, 255))

# For standalone testing
if __name__ == "__main__":
    print("Testing Mask Classifier...")
    
    # Create classifier
    classifier = MaskClassifier()
    
    # Test with sample image (if available)
    test_image_path = "test_face.jpg"
    if os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)
        is_proper, issues = classifier.check_proper_wearing(img)
        print(f"Proper wearing: {is_proper}")
        print(f"Issues: {issues}")
        
        mask_type, confidence, color = classifier.classify_mask_type(img, True)
        print(f"Mask type: {mask_type} (confidence: {confidence:.2f})")
    else:
        print("No test image found. Run with actual webcam for testing.")
    
    print("\n✅ Test complete")