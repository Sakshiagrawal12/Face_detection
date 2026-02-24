# utils/visualization.py
"""
Visualization Module
Provides plotting and visualization utilities for training and evaluation
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from config import COLORS_RGB, COLORS_BGR, OUTPUT_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Visualizer:
    """
    Visualization utilities for model training and evaluation
    
    Attributes:
        colors_rgb: RGB colors for matplotlib plots
        colors_bgr: BGR colors for OpenCV drawing
        save_format: Default image save format
        dpi: Resolution for saved figures
    """
    
    def __init__(self, save_format: str = 'png', dpi: int = 300):
        """
        Initialize visualizer
        
        Args:
            save_format: Image format for saved figures
            dpi: Resolution for saved figures
        """
        self.colors_rgb = COLORS_RGB
        self.colors_bgr = COLORS_BGR
        self.save_format = save_format
        self.dpi = dpi
        
        # Configure matplotlib style
        plt.style.use('default')
        logging.info(f"Visualizer initialized with {save_format} format, {dpi} DPI")
    
    def plot_training_history(self, history: Any, save_path: Optional[str] = None) -> None:
        """
        Plot training history (accuracy and loss)
        
        Args:
            history: Keras training history object
            save_path: Path to save the figure
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            ax1.plot(history.history.get('accuracy', []), 
                    label='Training Accuracy', linewidth=2, color='blue')
            ax1.plot(history.history.get('val_accuracy', []), 
                    label='Validation Accuracy', linewidth=2, color='orange')
            ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1])
            
            # Plot loss
            ax2.plot(history.history.get('loss', []), 
                    label='Training Loss', linewidth=2, color='blue')
            ax2.plot(history.history.get('val_loss', []), 
                    label='Validation Loss', linewidth=2, color='orange')
            ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logging.info(f"Training history saved to: {save_path}")
            
            plt.show()
            plt.close(fig)  # Free memory
            
        except Exception as e:
            logging.error(f"Error plotting training history: {e}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            save_path: Path to save the figure
        """
        try:
            plt.figure(figsize=(8, 6))
            
            # Plot confusion matrix
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.colorbar()
            
            # Add labels
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logging.info(f"Confusion matrix saved to: {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")
    
    def plot_class_distribution(self, class_counts: Dict[str, int], 
                               save_path: Optional[str] = None) -> None:
        """
        Plot class distribution bar chart
        
        Args:
            class_counts: Dictionary mapping class names to counts
            save_path: Path to save the figure
        """
        try:
            plt.figure(figsize=(10, 6))
            
            categories = list(class_counts.keys())
            counts = list(class_counts.values())
            
            # Use RGB colors for matplotlib
            colors = [
                self.colors_rgb.get('with_mask', (0, 1, 0)),    # Green
                self.colors_rgb.get('without_mask', (1, 0, 0))  # Red
            ]
            
            bars = plt.bar(categories, counts, color=colors, alpha=0.7)
            plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Number of Images', fontsize=12)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontsize=11)
            
            plt.grid(True, alpha=0.3, axis='y')
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logging.info(f"Class distribution saved to: {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting class distribution: {e}")
    
    def plot_sample_predictions(self, images: np.ndarray, true_labels: np.ndarray, 
                               pred_labels: np.ndarray, class_names: List[str], 
                               num_samples: int = 10, save_path: Optional[str] = None) -> None:
        """
        Plot sample images with predictions
        
        Args:
            images: Array of images
            true_labels: True labels
            pred_labels: Predicted labels
            class_names: List of class names
            num_samples: Number of samples to plot
            save_path: Path to save the figure
        """
        try:
            num_samples = min(num_samples, len(images))
            cols = 5
            rows = (num_samples + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
            if rows == 1:
                axes = [axes]
            axes = np.array(axes).flatten()
            
            # Randomly select samples
            indices = np.random.choice(len(images), num_samples, replace=False)
            
            for i, idx in enumerate(indices):
                # Display image
                if images[idx].max() <= 1.0:
                    img = (images[idx] * 255).astype(np.uint8)
                else:
                    img = images[idx].astype(np.uint8)
                
                axes[i].imshow(img)
                
                # Set title with color based on correctness
                true_name = class_names[true_labels[idx]]
                pred_name = class_names[pred_labels[idx]]
                
                if true_labels[idx] == pred_labels[idx]:
                    color = 'green'
                    title = f"✓ True: {true_name}\n  Pred: {pred_name}"
                else:
                    color = 'red'
                    title = f"✗ True: {true_name}\n  Pred: {pred_name}"
                
                axes[i].set_title(title, color=color, fontsize=8)
                axes[i].axis('off')
            
            # Hide empty subplots
            for i in range(num_samples, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logging.info(f"Sample predictions saved to: {save_path}")
            
            plt.show()
            plt.close(fig)
            
        except Exception as e:
            logging.error(f"Error plotting sample predictions: {e}")
    
    def draw_detection_results(self, frame: np.ndarray, faces: List[Tuple], 
                              predictions: List[int], confidences: List[float]) -> np.ndarray:
        """
        Draw detection results on frame (for webcam) using BGR colors
        
        Args:
            frame: Input frame
            faces: List of face rectangles (x, y, w, h)
            predictions: List of predictions (0=with_mask, 1=without_mask)
            confidences: List of confidence scores
            
        Returns:
            Frame with drawn detections
        """
        try:
            result = frame.copy()
            
            for (x, y, w, h), pred, conf in zip(faces, predictions, confidences):
                # Determine color based on prediction (using BGR colors)
                if pred == 0:  # With mask
                    color = self.colors_bgr.get('with_mask', (0, 255, 0))
                    label = f"Mask: {conf:.2%}"
                else:  # Without mask
                    color = self.colors_bgr.get('without_mask', (0, 0, 255))
                    label = f"No Mask: {conf:.2%}"
                
                # Draw bounding box
                cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    result,
                    (x, y - text_height - 10),
                    (x + text_width, y),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    result,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.colors_bgr.get('text', (255, 255, 255)),
                    2
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Error drawing detection results: {e}")
            return frame
    
    def create_dashboard(self, stats: Dict[str, Any], frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a dashboard with statistics overlay
        
        Args:
            stats: Dictionary of statistics to display
            frame: Optional base frame (creates blank if None)
            
        Returns:
            Dashboard image
        """
        try:
            if frame is None:
                # Create blank dashboard
                dashboard = np.ones((480, 640, 3), dtype=np.uint8) * 255
            else:
                dashboard = frame.copy()
            
            # Create semi-transparent overlay
            overlay = dashboard.copy()
            overlay_height = min(30 + len(stats) * 25, dashboard.shape[0] - 10)
            cv2.rectangle(overlay, (5, 5), (300, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, dashboard, 0.5, 0, dashboard)
            
            # Add statistics
            y_start = 30
            for i, (key, value) in enumerate(stats.items()):
                text = f"{key}: {value}"
                cv2.putText(
                    dashboard,
                    text,
                    (15, y_start + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.colors_bgr.get('text', (255, 255, 255)),
                    2
                )
            
            return dashboard
            
        except Exception as e:
            logging.error(f"Error creating dashboard: {e}")
            return frame if frame is not None else np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    def save_plot(self, fig: plt.Figure, save_path: str) -> None:
        """
        Save a matplotlib figure
        
        Args:
            fig: Matplotlib figure
            save_path: Path to save the figure
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logging.info(f"Plot saved to: {save_path}")
        except Exception as e:
            logging.error(f"Error saving plot: {e}")
    
    def close_all(self) -> None:
        """Close all matplotlib figures to free memory"""
        plt.close('all')
        logging.info("All plots closed")

# For standalone testing
if __name__ == "__main__":
    print("Testing Visualizer...")
    
    viz = Visualizer()
    
    # Test with sample data
    class_counts = {'with_mask': 1000, 'without_mask': 950}
    viz.plot_class_distribution(class_counts)
    
    print("\n✅ Test complete")