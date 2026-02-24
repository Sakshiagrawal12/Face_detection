# scripts/webcam_detection.py
"""
Webcam Detection Script with improved display size
Fixed FPS calculation and larger window for better visibility
Added: Hand detection, mask feature validation, unique face tracking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import tensorflow as tf
import time
import datetime
import uuid
from collections import deque
from config import BEST_MODEL_PATH, IMG_SIZE, SCREENSHOTS_PATH, CATEGORIES, COLORS_BGR

# Try to import mediapipe for hand detection (optional)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not installed. Hand detection disabled. Install with: pip install mediapipe")

class WebcamMaskDetector:
    def __init__(self, model_path=BEST_MODEL_PATH):
        """Initialize webcam detector with larger display"""
        print("\n" + "="*60)
        print("INITIALIZING WEBCAM DETECTOR")
        print("="*60)
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found at: {model_path}")
            print("Please train the model first: python main.py --mode train")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        print("📂 Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = IMG_SIZE
        print(f"✅ Model loaded successfully!")
        
        # Initialize face detector
        print("👤 Initializing face detector...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # ============================================
        # SOLUTION 3: Initialize hand detection
        # ============================================
        if MEDIAPIPE_AVAILABLE:
            print("🖐️ Initializing hand detector...")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hand_detection_available = True
            print("✅ Hand detector initialized")
        else:
            self.hand_detection_available = False
        
        # ============================================
        # SOLUTION 1: Unique face tracking
        # ============================================
        print("👥 Initializing face tracker...")
        self.face_tracker = {}  # Store unique faces
        self.next_face_id = 0
        self.unique_faces_count = 0
        self.unique_with_mask = 0
        self.unique_without_mask = 0
        self.face_history = {}  # Store prediction history per face
        self.tracking_timeout = 3.0  # Seconds before forgetting a face
        
        # Statistics (keeping your existing stats format)
        self.stats = {
            'total_faces': 0,  # Frame-by-frame count (for display)
            'with_mask': 0,    # Frame-by-frame count
            'without_mask': 0,  # Frame-by-frame count
            'unique_total': 0,  # Unique people count
            'unique_with_mask': 0,
            'unique_without_mask': 0
        }
        
        # For FPS calculation
        self.prev_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        # Create screenshots directory
        os.makedirs(SCREENSHOTS_PATH, exist_ok=True)
        
        print("✅ Detector ready!")
    
    # ============================================
    # SOLUTION 1: Face tracking methods
    # ============================================
    def get_face_id(self, face_box):
        """Assign unique ID to each face based on position"""
        x, y, w, h = face_box
        center = (x + w//2, y + h//2)
        current_time = time.time()
        
        # Clean up old faces (not seen for a while)
        expired = []
        for face_id, data in self.face_tracker.items():
            if current_time - data['last_seen'] > self.tracking_timeout:
                expired.append(face_id)
        
        for face_id in expired:
            del self.face_tracker[face_id]
            if face_id in self.face_history:
                del self.face_history[face_id]
        
        # Find closest existing face
        for face_id, data in self.face_tracker.items():
            last_center = data['last_center']
            distance = ((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)**0.5
            
            if distance < 100:  # Same person threshold
                # Update existing face
                data['last_center'] = center
                data['last_seen'] = current_time
                data['frame_count'] += 1
                return face_id
        
        # New face detected
        self.next_face_id += 1
        face_id = f"FACE_{self.next_face_id}"
        self.face_tracker[face_id] = {
            'last_center': center,
            'last_seen': current_time,
            'first_seen': current_time,
            'frame_count': 1
        }
        self.face_history[face_id] = deque(maxlen=15)  # Store last 15 predictions
        self.unique_faces_count += 1
        
        return face_id
    
    def update_face_history(self, face_id, has_mask, confidence):
        """Update prediction history for a face"""
        if face_id in self.face_history:
            self.face_history[face_id].append({
                'has_mask': has_mask,
                'confidence': confidence,
                'time': time.time()
            })
    
    def get_face_consensus(self, face_id):
        """Get consensus prediction from history"""
        if face_id not in self.face_history or len(self.face_history[face_id]) < 5:
            return None, 0
        
        history = list(self.face_history[face_id])
        mask_count = sum(1 for h in history if h['has_mask'])
        total = len(history)
        
        if mask_count / total > 0.7:  # 70% of frames say mask
            return "MASK", mask_count / total
        elif (total - mask_count) / total > 0.7:  # 70% say no mask
            return "NO_MASK", (total - mask_count) / total
        else:
            return "UNCERTAIN", max(mask_count, total - mask_count) / total
    
    # ============================================
    # SOLUTION 3: Hand detection
    # ============================================
    def is_hand_covering_face(self, face_roi):
        """Detect if hand is covering the face"""
        if not self.hand_detection_available:
            return False
        
        try:
            # Convert to RGB for MediaPipe
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_face)
            
            if results.multi_hand_landmarks:
                # Hands detected in face region
                return True
            
            # Alternative: Check for skin-like colors in lower face
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            # Skin color range in HSV
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 150, 255])
            
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
            
            # High skin percentage in lower face might indicate hand
            if skin_percentage > 0.4:
                return True
            
            return False
            
        except Exception as e:
            return False
    
    # ============================================
    # SOLUTION 4: Mask feature validation
    # ============================================
    def has_mask_features(self, face_roi):
        """Validate that the detected object has mask-like features"""
        try:
            h, w = face_roi.shape[:2]
            
            # Focus on lower half of face where mask should be
            lower_face = face_roi[h//2:, :]
            
            if lower_face.size == 0:
                return False, 0
            
            # Feature 1: Color consistency (masks have uniform color)
            hsv_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
            color_std = np.std(hsv_lower[:, :, 1])  # Saturation standard deviation
            
            # Hands have high color variation, masks have low variation
            color_consistent = color_std < 40
            
            # Feature 2: Edge density (masks have smooth edges)
            gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_lower, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Hands create many edges, masks have fewer edges
            smooth_surface = edge_density < 0.15
            
            # Feature 3: Check for mask straps (simplified)
            # Look for dark lines at the sides of face
            left_side = lower_face[:, :w//8]
            right_side = lower_face[:, -w//8:]
            
            left_brightness = np.mean(cv2.cvtColor(left_side, cv2.COLOR_BGR2GRAY))
            right_brightness = np.mean(cv2.cvtColor(right_side, cv2.COLOR_BGR2GRAY))
            
            # Straps are usually darker than face
            has_straps = (left_brightness < 100 and right_brightness < 100)
            
            # Feature 4: Check for mask shape (horizontal line in middle)
            middle_line = lower_face[h//4:3*h//4, :]
            gray_middle = cv2.cvtColor(middle_line, cv2.COLOR_BGR2GRAY)
            horizontal_edges = cv2.Sobel(gray_middle, cv2.CV_64F, 1, 0, ksize=3)
            strong_horizontal = np.mean(np.abs(horizontal_edges)) > 20
            
            # Combine features
            feature_score = 0
            if color_consistent:
                feature_score += 25
            if smooth_surface:
                feature_score += 25
            if has_straps:
                feature_score += 25
            if strong_horizontal:
                feature_score += 25
            
            is_mask_like = feature_score >= 50  # At least 2 features
            
            return is_mask_like, feature_score
            
        except Exception as e:
            return True, 50  # Default to true on error
    
    def preprocess_face(self, face_roi):
        """Preprocess face for model input"""
        try:
            face = cv2.resize(face_roi, (self.input_size, self.input_size))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=0)
            return face
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def run(self):
        """Main detection loop with larger display"""
        print("\n🎥 Starting webcam...")
        print("Controls:")
        print("  • 'q' - Quit")
        print("  • 's' - Save screenshot")
        print("  • 'r' - Reset stats")
        print("-"*40)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # ============================================
        # SET LARGER RESOLUTION FOR BETTER DISPLAY
        # ============================================
        # Try to set larger resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Check what resolution was actually set
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"✅ Webcam resolution: {actual_width}x{actual_height} at {actual_fps:.1f} FPS")
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # ============================================
        # CREATE RESIZABLE WINDOW (DO THIS ONCE)
        # ============================================
        cv2.namedWindow('Face Mask Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Mask Detection', 1280, 720)
        
        # For FPS calculation
        self.prev_time = time.time()
        frame_count = 0
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            frame_count += 1
            
            # ============================================
            # SAFE FPS CALCULATION
            # ============================================
            current_time = time.time()
            time_diff = current_time - self.prev_time
            if time_diff > 0.001:
                self.fps = 1.0 / time_diff
            self.prev_time = current_time
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )
            
            # Track which faces we've processed this frame
            processed_face_ids = []
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face ROI with small margin
                margin = 10
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size == 0:
                    continue
                
                # ============================================
                # SOLUTION 1: Get unique face ID
                # ============================================
                face_id = self.get_face_id((x, y, w, h))
                processed_face_ids.append(face_id)
                
                # Preprocess and predict
                processed_face = self.preprocess_face(face_roi)
                if processed_face is None:
                    continue
                    
                predictions = self.model.predict(processed_face, verbose=0)[0]
                
                # Get class and confidence
                class_idx = np.argmax(predictions)
                confidence = predictions[class_idx]
                model_says_mask = (class_idx == 0)
                
                # ============================================
                # SOLUTION 3: Check for hands covering face
                # ============================================
                hand_detected = self.is_hand_covering_face(face_roi)
                
                # ============================================
                # SOLUTION 4: Validate mask features
                # ============================================
                has_mask_features, feature_score = self.has_mask_features(face_roi)
                
                # ============================================
                # Combine all signals for final decision
                # ============================================
                is_actually_mask = False
                final_decision = "No Mask"
                
                if model_says_mask:
                    if hand_detected:
                        # Hand covering face - definitely not a mask
                        is_actually_mask = False
                        final_decision = "No Mask (Hand)"
                    elif has_mask_features:
                        # Model says mask AND has mask features - likely correct
                        is_actually_mask = True
                        final_decision = "Mask"
                    else:
                        # Model says mask but no mask features - could be false positive
                        is_actually_mask = False
                        final_decision = "Uncertain"
                else:
                    # Model says no mask
                    is_actually_mask = False
                    final_decision = "No Mask"
                
                # ============================================
                # Update face history and get consensus
                # ============================================
                self.update_face_history(face_id, is_actually_mask, confidence)
                consensus, consensus_conf = self.get_face_consensus(face_id)
                
                if consensus == "MASK":
                    final_decision = "Mask"
                    is_actually_mask = True
                elif consensus == "NO_MASK":
                    final_decision = "No Mask"
                    is_actually_mask = False
                
                # ============================================
                # Update statistics (keeping your existing format)
                # ============================================
                self.stats['total_faces'] += 1
                
                if is_actually_mask:
                    self.stats['with_mask'] += 1
                else:
                    self.stats['without_mask'] += 1
                
                # Update unique face statistics (only count each face once)
                if face_id not in [f'processed_{pid}' for pid in processed_face_ids[:-1]]:
                    if is_actually_mask:
                        self.stats['unique_with_mask'] += 1
                    else:
                        self.stats['unique_without_mask'] += 1
                    self.stats['unique_total'] = self.stats['unique_with_mask'] + self.stats['unique_without_mask']
                
                # ============================================
                # Determine box color based on decision
                # ============================================
                if final_decision == "Mask":
                    color = COLORS_BGR.get('with_mask', (0, 255, 0))
                    label = f"Mask: {confidence:.2%}"
                elif final_decision == "No Mask (Hand)":
                    color = (0, 165, 255)  # Orange
                    label = f"Hand detected"
                elif final_decision == "Uncertain":
                    color = (0, 255, 255)  # Yellow
                    label = f"Uncertain: {confidence:.2%}"
                else:
                    color = COLORS_BGR.get('without_mask', (0, 0, 255))
                    label = f"No Mask: {confidence:.2%}"
                
                # Add feature score to label for debugging (optional)
                # label += f" F:{feature_score}"
                
                # ============================================
                # DRAW BOUNDING BOX
                # ============================================
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    display_frame,
                    (x, y - text_height - 15),
                    (x + text_width + 10, y),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    display_frame,
                    label,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Draw face ID (optional, for debugging)
                # cv2.putText(
                #     display_frame,
                #     face_id[-4:],
                #     (x, y + h + 20),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (200, 200, 200),
                #     1
                # )
            
            # ============================================
            # DRAW STATISTICS PANEL (UPDATED WITH UNIQUE COUNTS)
            # ============================================
            self.draw_stats_panel(display_frame)
            
            # ============================================
            # DRAW INSTRUCTIONS PANEL
            # ============================================
            self.draw_instructions(display_frame)
            
            # ============================================
            # ADD TIMESTAMP AND FPS
            # ============================================
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                display_frame,
                timestamp,
                (display_frame.shape[1] - 300, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Add FPS display
            cv2.putText(
                display_frame,
                f"FPS: {self.fps:.1f}",
                (display_frame.shape[1] - 150, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow('Face Mask Detection', display_frame)
            
            # ============================================
            # HANDLE KEY PRESSES
            # ============================================
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                self.save_screenshot(display_frame)
            elif key == ord('r') or key == ord('R'):
                self.reset_stats()
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        self.print_final_stats()
    
    def draw_stats_panel(self, frame):
        """Draw statistics panel on frame (updated with unique counts)"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay (larger panel)
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (450, 280), (0, 0, 0), -1)  # Made panel wider
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw statistics with larger text
        y_offset = 50
        stats_text = [
            f"FRAME STATS (this session):",
            f"  Total Detections: {self.stats['total_faces']}",
            f"  With Mask: {self.stats['with_mask']}",
            f"  Without Mask: {self.stats['without_mask']}",
            f"",
            f"UNIQUE PEOPLE (counted once):",
            f"  Total People: {self.stats['unique_total']}",
            f"  With Mask: {self.stats['unique_with_mask']}",
            f"  Without Mask: {self.stats['unique_without_mask']}"
        ]
        
        if self.stats['unique_total'] > 0:
            compliance = (self.stats['unique_with_mask'] / self.stats['unique_total']) * 100
            stats_text.append(f"  Compliance: {compliance:.1f}%")
        
        for i, text in enumerate(stats_text):
            if text == "":
                y_offset += 10
                continue
            
            if "FRAME STATS" in text or "UNIQUE PEOPLE" in text:
                # Section headers in cyan
                cv2.putText(
                    frame,
                    text,
                    (40, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    text,
                    (40, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
            y_offset += 25
    
    def draw_instructions(self, frame):
        """Draw instructions on frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for instructions
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 300, h - 150), (w - 20, h - 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        instructions = [
            "Controls:",
            "Q: Quit",
            "S: Save Screenshot",
            "R: Reset Stats"
        ]
        
        y_start = h - 120
        for i, text in enumerate(instructions):
            if i == 0:
                # Title in different color
                cv2.putText(
                    frame,
                    text,
                    (w - 280, y_start + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    text,
                    (w - 280, y_start + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1
                )
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mask_detection_{timestamp}.jpg"
        filepath = os.path.join(SCREENSHOTS_PATH, filename)
        cv2.imwrite(filepath, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_faces': 0,
            'with_mask': 0,
            'without_mask': 0,
            'unique_total': 0,
            'unique_with_mask': 0,
            'unique_without_mask': 0
        }
        self.face_tracker = {}
        self.face_history = {}
        self.next_face_id = 0
        self.unique_faces_count = 0
        print("📊 Statistics reset")
    
    def print_final_stats(self):
        """Print final statistics"""
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"\n📊 FRAME-BY-FRAME STATS:")
        print(f"  Total detections: {self.stats['total_faces']}")
        print(f"  With mask: {self.stats['with_mask']}")
        print(f"  Without mask: {self.stats['without_mask']}")
        
        print(f"\n👥 UNIQUE PEOPLE STATS:")
        print(f"  Total unique people: {self.stats['unique_total']}")
        print(f"  With mask: {self.stats['unique_with_mask']}")
        print(f"  Without mask: {self.stats['unique_without_mask']}")
        
        if self.stats['unique_total'] > 0:
            compliance = (self.stats['unique_with_mask'] / self.stats['unique_total']) * 100
            print(f"\n📈 True compliance rate: {compliance:.1f}%")

# ============================================
# MAIN ENTRY POINT
# ============================================
if __name__ == "__main__":
    try:
        detector = WebcamMaskDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()