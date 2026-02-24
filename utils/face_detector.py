# utils/face_detector.py
"""
Face Detection Module
Provides face detection using multiple Haar cascades with performance optimizations
"""

import cv2
import numpy as np
import os
import logging
from config import MIN_FACE_SIZE, FACE_DETECTION_CONFIDENCE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceDetector:
    """
    Face detector using OpenCV Haar Cascades with multiple fallback options
    
    Attributes:
        face_cascades (list): List of loaded cascade classifiers
        min_face_size (tuple): Minimum face size to detect (w, h)
        confidence_threshold (float): Minimum confidence for detection
        frame_skip (int): Process every Nth frame for performance
    """
    
    def __init__(self, min_face_size=MIN_FACE_SIZE, confidence_threshold=0.5, frame_skip=2):
        """
        Initialize face detector with multiple cascade classifiers
        
        Args:
            min_face_size (tuple): Minimum face size (width, height)
            confidence_threshold (float): Minimum confidence for detection
            frame_skip (int): Process every Nth frame (1 = all frames)
        """
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.frame_skip = frame_skip
        self.frame_count = 0
        
        # Try multiple cascade files with fallback options
        cascade_files = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt_tree.xml'
        ]
        
        self.face_cascades = []
        
        # Try loading from multiple possible locations
        search_paths = [
            cv2.data.haarcascades,  # OpenCV installation
            'cascades/',             # Local cascades folder
            './',                     # Current directory
            os.path.join(os.path.dirname(__file__), '..', 'cascades')  # Project cascades
        ]
        
        for cascade_file in cascade_files:
            for base_path in search_paths:
                cascade_path = os.path.join(base_path, cascade_file)
                if os.path.exists(cascade_path):
                    try:
                        cascade = cv2.CascadeClassifier(cascade_path)
                        if not cascade.empty():
                            self.face_cascades.append(cascade)
                            logging.info(f"✅ Loaded cascade: {cascade_path}")
                            break
                    except Exception as e:
                        logging.warning(f"Failed to load {cascade_path}: {e}")
        
        if not self.face_cascades:
            logging.error("No face detection cascades could be loaded!")
            raise ValueError("Could not load any face detection cascades. Please download Haar cascade files.")
        
        logging.info(f"✅ FaceDetector initialized with {len(self.face_cascades)} cascade(s)")
        
    def detect_faces(self, image, return_confidences=False):
        """
        Detect faces in an image with confidence scores
        
        Args:
            image (numpy.ndarray): Input image
            return_confidences (bool): Whether to return confidence scores
            
        Returns:
            list: List of (x, y, w, h) face rectangles
            or list of (x, y, w, h, confidence) if return_confidences=True
        """
        if image is None:
            return [] if not return_confidences else []
        
        # Frame skipping for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return [] if not return_confidences else []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Enhance image for better detection
            gray = cv2.equalizeHist(gray)
            
            # Resize for faster processing if image is large
            scale_factor = 1.0
            if gray.shape[1] > 640:
                scale_factor = 640 / gray.shape[1]
                new_width = int(gray.shape[1] * scale_factor)
                new_height = int(gray.shape[0] * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height))
            
            all_faces = []
            all_confidences = []
            
            # Try different cascades
            for cascade in self.face_cascades:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=self.min_face_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) > 0:
                    # Scale faces back to original image size
                    if scale_factor != 1.0:
                        faces = [(int(x/scale_factor), int(y/scale_factor), 
                                 int(w/scale_factor), int(h/scale_factor)) 
                                for (x, y, w, h) in faces]
                    
                    # Calculate confidence based on detection parameters
                    for (x, y, w, h) in faces:
                        # Simple confidence heuristic: larger faces = higher confidence
                        face_area = w * h
                        confidence = min(1.0, face_area / (100 * 100))  # Normalize
                        confidence = max(self.confidence_threshold, confidence)
                        
                        all_faces.append((x, y, w, h))
                        all_confidences.append(confidence)
            
            # Remove duplicates
            if len(all_faces) > 1:
                all_faces, all_confidences = self._remove_overlapping_faces(
                    all_faces, all_confidences
                )
            
            # Filter by confidence
            filtered_results = []
            for i, (face, conf) in enumerate(zip(all_faces, all_confidences)):
                if conf >= self.confidence_threshold:
                    if return_confidences:
                        filtered_results.append((*face, conf))
                    else:
                        filtered_results.append(face)
            
            return filtered_results
            
        except Exception as e:
            logging.error(f"Error in face detection: {e}")
            return [] if not return_confidences else []
    
    def _remove_overlapping_faces(self, faces, confidences, overlap_threshold=0.3):
        """
        Remove overlapping face detections using Non-Maximum Suppression
        
        Args:
            faces (list): List of face rectangles
            confidences (list): List of confidence scores
            overlap_threshold (float): IoU threshold for suppression
            
        Returns:
            tuple: Filtered faces and confidences
        """
        if len(faces) <= 1:
            return faces, confidences
        
        # Convert to list of rectangles (x1, y1, x2, y2)
        rects = [(x, y, x+w, y+h) for (x, y, w, h) in faces]
        
        # Calculate areas
        areas = [(x2-x1)*(y2-y1) for (x1, y1, x2, y2) in rects]
        
        # Sort by confidence (highest first)
        indices = sorted(range(len(confidences)), 
                        key=lambda i: confidences[i], 
                        reverse=True)
        
        keep_indices = []
        
        while indices:
            current = indices.pop(0)
            keep_indices.append(current)
            
            to_remove = []
            for i in indices:
                # Calculate IoU (Intersection over Union)
                x1 = max(rects[current][0], rects[i][0])
                y1 = max(rects[current][1], rects[i][1])
                x2 = min(rects[current][2], rects[i][2])
                y2 = min(rects[current][3], rects[i][3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = areas[current] + areas[i] - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > overlap_threshold:
                        to_remove.append(i)
            
            indices = [i for i in indices if i not in to_remove]
        
        # Return filtered faces and confidences
        filtered_faces = [faces[i] for i in keep_indices]
        filtered_confidences = [confidences[i] for i in keep_indices]
        
        return filtered_faces, filtered_confidences
    
    def get_face_landmarks(self, image, face):
        """
        Get facial landmarks (simplified version)
        Note: For full landmarks, use dlib or mediapipe
        
        Args:
            image (numpy.ndarray): Input image
            face (tuple): Face rectangle (x, y, w, h)
            
        Returns:
            dict: Dictionary of facial feature regions
        """
        x, y, w, h = face
        face_roi = image[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return {}
        
        # Approximate facial feature regions (simplified)
        landmarks = {
            'left_eye': (x + w//4, y + h//4),
            'right_eye': (x + 3*w//4, y + h//4),
            'nose': (x + w//2, y + h//2),
            'mouth': (x + w//2, y + 3*h//4)
        }
        
        return landmarks
    
    def extract_face_roi(self, image, face, margin=0.2):
        """
        Extract face region of interest with margin
        
        Args:
            image (numpy.ndarray): Input image
            face (tuple): Face rectangle (x, y, w, h)
            margin (float): Margin to add around face (0.2 = 20%)
            
        Returns:
            numpy.ndarray: Face ROI
        """
        x, y, w, h = face
        
        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        return image[y1:y2, x1:x2]
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2, draw_confidence=False):
        """
        Draw rectangles around detected faces with optional confidence
        
        Args:
            image (numpy.ndarray): Input image
            faces (list): List of face rectangles or (rect, confidence) tuples
            color (tuple): BGR color for drawing
            thickness (int): Line thickness
            draw_confidence (bool): Whether to draw confidence scores
            
        Returns:
            numpy.ndarray: Image with drawn faces
        """
        result = image.copy()
        
        for face_data in faces:
            if len(face_data) == 4:
                x, y, w, h = face_data
                confidence = None
            else:
                x, y, w, h, confidence = face_data
            
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
            
            # Draw confidence if available
            if draw_confidence and confidence is not None:
                label = f"{confidence:.2f}"
                cv2.putText(result, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def validate_dataset_faces(self, dataset_path, categories):
        """
        Validate that images in dataset contain faces
        
        Args:
            dataset_path (str): Path to dataset
            categories (list): List of category folders
            
        Returns:
            list: Paths to images with no faces detected
        """
        from tqdm import tqdm
        import os
        
        no_face_images = []
        
        for category in categories:
            category_path = os.path.join(dataset_path, category)
            if not os.path.exists(category_path):
                logging.warning(f"Category path not found: {category_path}")
                continue
            
            print(f"\n🔍 Checking faces in {category}...")
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for img_name in tqdm(image_files, desc=f"Processing {category}"):
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    faces = self.detect_faces(img)
                    if len(faces) == 0:
                        no_face_images.append(img_path)
                        logging.debug(f"No face detected: {img_name}")
        
        logging.info(f"Found {len(no_face_images)} images with no faces")
        return no_face_images
    
    def get_optimal_frame_skip(self, target_fps=30):
        """
        Calculate optimal frame skip for target FPS
        
        Args:
            target_fps (int): Desired processing FPS
            
        Returns:
            int: Recommended frame skip value
        """
        # Assuming detection takes ~0.03s per frame
        detection_time = 0.03
        optimal_skip = max(1, int(target_fps * detection_time))
        return min(optimal_skip, 5)  # Cap at 5

# For standalone testing
if __name__ == "__main__":
    print("Testing Face Detector...")
    
    try:
        detector = FaceDetector(frame_skip=1)
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            exit()
        
        print("✅ Webcam opened. Press 'q' to quit.")
        print("Press 'c' to toggle confidence display")
        
        show_confidence = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces with confidence
            faces = detector.detect_faces(frame, return_confidences=True)
            
            # Draw results
            frame = detector.draw_faces(frame, faces, draw_confidence=show_confidence)
            
            # Add info text
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Face Detector Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_confidence = not show_confidence
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()