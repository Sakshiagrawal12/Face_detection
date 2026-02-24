# utils/face_tracker.py
"""
Multi-Face Tracking Module
Tracks multiple faces across video frames with persistent IDs and history
"""

import uuid
import time
import math
import cv2
import numpy as np
from collections import deque, Counter
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceTracker:
    """
    FEATURE 1: Multi-Face Tracking with History
    Tracks multiple faces across video frames with unique IDs
    
    Attributes:
        tracked_faces (dict): Dictionary of tracked face data
        tracking_timeout (float): Seconds before forgetting a face
        max_history (int): Maximum history entries per face
        colors (dict): Color assignments for each track ID
        next_id (int): Counter for generating sequential IDs
        frame_size (tuple): Expected frame size for normalization
    """
    
    def __init__(self, 
                 tracking_timeout: float = 3.0, 
                 max_history: int = 30,
                 save_path: Optional[str] = None,
                 frame_size: Tuple[int, int] = (640, 480)):
        """
        Initialize face tracker
        
        Args:
            tracking_timeout: Seconds before forgetting a face
            max_history: Maximum history entries per face
            save_path: Path to save/load tracking data
            frame_size: Expected frame size for normalization
        """
        self.tracked_faces: Dict[str, dict] = {}
        self.tracking_timeout = tracking_timeout
        self.max_history = max_history
        self.frame_size = frame_size
        self.colors: Dict[str, tuple] = {}
        self.next_id = 1
        self.save_path = save_path
        
        # Performance metrics
        self.total_tracks_created = 0
        self.total_tracks_expired = 0
        
        # Load previous tracking data if available
        if save_path and os.path.exists(save_path):
            self.load_tracks(save_path)
        
        logging.info(f"FaceTracker initialized with timeout={tracking_timeout}s, max_history={max_history}")
    
    def calculate_distance(self, face1: Tuple[int, int, int, int], 
                          face2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Euclidean distance between two face positions
        
        Args:
            face1: First face rectangle (x, y, w, h)
            face2: Second face rectangle (x, y, w, h)
            
        Returns:
            float: Distance between face centers
        """
        try:
            x1, y1, w1, h1 = face1
            x2, y2, w2, h2 = face2
            
            # Use center points for comparison
            center1 = (x1 + w1//2, y1 + h1//2)
            center2 = (x2 + w2//2, y2 + h2//2)
            
            distance = math.sqrt((center1[0] - center2[0])**2 + 
                                (center1[1] - center2[1])**2)
            return distance
        except Exception as e:
            logging.error(f"Error calculating distance: {e}")
            return float('inf')
    
    def calculate_iou(self, face1: Tuple[int, int, int, int], 
                      face2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union between two faces
        
        Args:
            face1: First face rectangle
            face2: Second face rectangle
            
        Returns:
            float: IoU score between 0 and 1
        """
        try:
            x1, y1, w1, h1 = face1
            x2, y2, w2, h2 = face2
            
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating IoU: {e}")
            return 0.0
    
    def get_color_for_id(self, track_id: str) -> tuple:
        """
        Assign a consistent color to each tracked ID
        
        Args:
            track_id: Track identifier
            
        Returns:
            tuple: BGR color tuple
        """
        if track_id not in self.colors:
            # Generate color based on ID hash
            hash_val = hash(track_id) % 1000
            np.random.seed(hash_val)
            self.colors[track_id] = tuple(map(int, np.random.randint(100, 255, 3)))
        return self.colors[track_id]
    
    def generate_id(self) -> str:
        """
        Generate a unique ID for a new face
        
        Returns:
            str: Unique track ID
        """
        track_id = f"P{self.next_id:04d}"
        self.next_id += 1
        self.total_tracks_created += 1
        return track_id
    
    def update_tracks(self, detected_faces: List[Tuple[int, int, int, int]], 
                     face_data_list: List[Dict[str, Any]]) -> Dict[str, dict]:
        """
        Update tracked faces with new detections
        
        Args:
            detected_faces: List of (x, y, w, h) from face detector
            face_data_list: List of dict with mask_type, confidence, compliance
            
        Returns:
            dict: Updated tracked faces
        """
        try:
            current_time = time.time()
            updated_tracks = {}
            
            # Pad face_data_list if necessary
            while len(face_data_list) < len(detected_faces):
                face_data_list.append({})
            
            # Match new detections with existing tracks
            matched = set()
            
            for i, face in enumerate(detected_faces):
                face_data = face_data_list[i]
                
                best_match = None
                best_score = 0
                
                # Find best matching existing track
                for track_id, track_data in self.tracked_faces.items():
                    if track_id in matched:
                        continue
                    
                    # Calculate matching score (combination of distance and IoU)
                    distance = self.calculate_distance(face, track_data['last_position'])
                    iou = self.calculate_iou(face, track_data['last_position'])
                    
                    # Normalize distance score (closer = higher score)
                    distance_score = max(0, 1 - (distance / 200))
                    
                    # Combined score (weighted)
                    score = 0.7 * iou + 0.3 * distance_score
                    
                    if score > 0.5 and score > best_score:
                        best_score = score
                        best_match = track_id
                
                if best_match is not None:
                    # Update existing track
                    track = self.tracked_faces[best_match]
                    track['last_position'] = face
                    track['last_seen'] = current_time
                    track['detection_count'] = track.get('detection_count', 0) + 1
                    track['match_score'] = best_score
                    
                    # Update mask history (using deque for efficiency)
                    if 'mask_type' in face_data and face_data['mask_type']:
                        if 'mask_history' not in track:
                            track['mask_history'] = deque(maxlen=self.max_history)
                        track['mask_history'].append(face_data['mask_type'])
                    
                    if 'compliance' in face_data:
                        if 'compliance_history' not in track:
                            track['compliance_history'] = deque(maxlen=self.max_history)
                        track['compliance_history'].append(face_data['compliance'])
                    
                    # Calculate moving averages safely
                    track['avg_compliance'] = self._safe_average(
                        track.get('compliance_history', [])
                    )
                    
                    track['dominant_mask'] = self._get_dominant_mask(
                        track.get('mask_history', [])
                    )
                    
                    updated_tracks[best_match] = track
                    matched.add(best_match)
                else:
                    # Create new track
                    track_id = self.generate_id()
                    updated_tracks[track_id] = {
                        'id': track_id,
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'last_position': face,
                        'detection_count': 1,
                        'match_score': 1.0,
                        'mask_history': deque(
                            [face_data.get('mask_type', 'unknown')] if face_data.get('mask_type') else [],
                            maxlen=self.max_history
                        ),
                        'compliance_history': deque(
                            [face_data.get('compliance', 0)] if face_data.get('compliance') is not None else [],
                            maxlen=self.max_history
                        ),
                        'avg_compliance': face_data.get('compliance', 0),
                        'dominant_mask': face_data.get('mask_type', 'unknown'),
                        'color': self.get_color_for_id(track_id)
                    }
            
            # Keep tracks that are still active (not timed out)
            expired_tracks = []
            for track_id, track_data in self.tracked_faces.items():
                if track_id not in updated_tracks:
                    if current_time - track_data['last_seen'] < self.tracking_timeout:
                        updated_tracks[track_id] = track_data
                    else:
                        expired_tracks.append(track_id)
            
            self.total_tracks_expired += len(expired_tracks)
            self.tracked_faces = updated_tracks
            
            # Auto-save if path provided
            if self.save_path and len(self.tracked_faces) % 10 == 0:
                self.save_tracks(self.save_path)
            
            return self.tracked_faces
            
        except Exception as e:
            logging.error(f"Error updating tracks: {e}")
            return self.tracked_faces
    
    def _safe_average(self, history: deque) -> float:
        """
        Safely calculate average of history
        
        Args:
            history: Deque of values
            
        Returns:
            float: Average value or 0 if empty
        """
        if not history:
            return 0.0
        try:
            return sum(history) / len(history)
        except:
            return 0.0
    
    def _get_dominant_mask(self, history: deque) -> str:
        """
        Get dominant mask type from history
        
        Args:
            history: Deque of mask types
            
        Returns:
            str: Most common mask type
        """
        if not history:
            return 'unknown'
        try:
            counter = Counter(history)
            return counter.most_common(1)[0][0]
        except:
            return 'unknown'
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall tracking statistics
        
        Returns:
            dict: Tracking statistics
        """
        try:
            current_time = time.time()
            
            # Count currently present faces
            currently_present = 0
            total_compliance = 0
            mask_distribution = {
                'surgical': 0, 'n95': 0, 'cloth': 0, 
                'improper': 0, 'proper': 0, 'below_nose': 0,
                'unknown': 0
            }
            
            for track in self.tracked_faces.values():
                if current_time - track['last_seen'] < 1.0:
                    currently_present += 1
                
                total_compliance += track.get('avg_compliance', 0)
                
                mask_type = track.get('dominant_mask', 'unknown')
                if mask_type in mask_distribution:
                    mask_distribution[mask_type] += 1
                else:
                    mask_distribution['unknown'] += 1
            
            stats = {
                'total_people_seen': len(self.tracked_faces),
                'currently_present': currently_present,
                'avg_compliance': total_compliance / len(self.tracked_faces) if self.tracked_faces else 0,
                'mask_distribution': mask_distribution,
                'total_tracks_created': self.total_tracks_created,
                'total_tracks_expired': self.total_tracks_expired
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting statistics: {e}")
            return {
                'total_people_seen': 0,
                'currently_present': 0,
                'avg_compliance': 0,
                'mask_distribution': {},
                'total_tracks_created': self.total_tracks_created,
                'total_tracks_expired': self.total_tracks_expired
            }
    
    def get_track_history(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full history for a specific track
        
        Args:
            track_id: Track identifier
            
        Returns:
            dict: Track history or None if not found
        """
        if track_id in self.tracked_faces:
            track = self.tracked_faces[track_id].copy()
            # Convert deques to lists for serialization
            if 'mask_history' in track:
                track['mask_history'] = list(track['mask_history'])
            if 'compliance_history' in track:
                track['compliance_history'] = list(track['compliance_history'])
            return track
        return None
    
    def save_tracks(self, filepath: str):
        """
        Save tracking data to file
        
        Args:
            filepath: Path to save file
        """
        try:
            # Convert deques to lists for JSON serialization
            serializable_tracks = {}
            for track_id, track in self.tracked_faces.items():
                serializable_track = track.copy()
                if 'mask_history' in serializable_track:
                    serializable_track['mask_history'] = list(serializable_track['mask_history'])
                if 'compliance_history' in serializable_track:
                    serializable_track['compliance_history'] = list(serializable_track['compliance_history'])
                serializable_tracks[track_id] = serializable_track
            
            data = {
                'tracks': serializable_tracks,
                'next_id': self.next_id,
                'total_created': self.total_tracks_created,
                'total_expired': self.total_tracks_expired,
                'timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logging.info(f"Saved {len(self.tracked_faces)} tracks to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving tracks: {e}")
    
    def load_tracks(self, filepath: str):
        """
        Load tracking data from file
        
        Args:
            filepath: Path to load file
        """
        try:
            if not os.path.exists(filepath):
                return
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to deques
            self.tracked_faces = {}
            for track_id, track in data.get('tracks', {}).items():
                if 'mask_history' in track:
                    track['mask_history'] = deque(track['mask_history'], maxlen=self.max_history)
                if 'compliance_history' in track:
                    track['compliance_history'] = deque(track['compliance_history'], maxlen=self.max_history)
                self.tracked_faces[track_id] = track
            
            self.next_id = data.get('next_id', 1)
            self.total_tracks_created = data.get('total_created', 0)
            self.total_tracks_expired = data.get('total_expired', 0)
            
            logging.info(f"Loaded {len(self.tracked_faces)} tracks from {filepath}")
            
        except Exception as e:
            logging.error(f"Error loading tracks: {e}")
    
    def reset(self):
        """Reset tracker to initial state"""
        self.tracked_faces.clear()
        self.colors.clear()
        self.next_id = 1
        self.total_tracks_created = 0
        self.total_tracks_expired = 0
        logging.info("Tracker reset")
    
    def cleanup_old_tracks(self, max_age: float = 10.0):
        """
        Force cleanup of tracks older than max_age
        
        Args:
            max_age: Maximum age in seconds
        """
        current_time = time.time()
        expired = []
        
        for track_id, track in self.tracked_faces.items():
            if current_time - track['last_seen'] > max_age:
                expired.append(track_id)
        
        for track_id in expired:
            del self.tracked_faces[track_id]
            if track_id in self.colors:
                del self.colors[track_id]
        
        self.total_tracks_expired += len(expired)
        
        if expired:
            logging.info(f"Cleaned up {len(expired)} old tracks")

# For standalone testing
if __name__ == "__main__":
    print("Testing Face Tracker...")
    
    # Create tracker
    tracker = FaceTracker(tracking_timeout=2.0, max_history=20)
    
    # Simulate some detections
    test_faces = [
        (100, 100, 50, 50),  # Face 1
        (300, 150, 60, 60),  # Face 2
    ]
    
    test_data = [
        {'mask_type': 'surgical', 'compliance': 95},
        {'mask_type': 'cloth', 'compliance': 80}
    ]
    
    # Update tracks
    tracks = tracker.update_tracks(test_faces, test_data)
    
    print(f"\nTracked {len(tracks)} faces:")
    for track_id, track in tracks.items():
        print(f"  {track_id}: {track['dominant_mask']} (compliance: {track['avg_compliance']:.1f}%)")
    
    # Get statistics
    stats = tracker.get_statistics()
    print(f"\nStatistics: {stats}")
    
    print("\n✅ Test complete")