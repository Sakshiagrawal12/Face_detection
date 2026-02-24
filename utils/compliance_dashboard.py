# utils/compliance_dashboard.py
import cv2
import numpy as np
import time
from collections import deque

class ComplianceDashboard:
    """
    FEATURE 2: Mask Compliance Score with Visual Indicators
    Creates a real-time dashboard showing compliance metrics
    """
    
    def __init__(self, history_size=100):
        self.compliance_history = deque(maxlen=history_size)
        self.start_time = time.time()
        self.frame_count = 0
        
    def get_compliance_color(self, score):
        """Get color based on compliance score"""
        if score >= 95:
            return (0, 255, 0)      # Green - Excellent
        elif score >= 80:
            return (0, 255, 255)    # Yellow - Good
        else:
            return (0, 0, 255)      # Red - Poor
    
    def draw_dashboard(self, frame, tracker_stats, current_compliance):
        """
        Draw comprehensive compliance dashboard on frame
        """
        h, w = frame.shape[:2]
        
        # Update compliance history
        self.compliance_history.append(current_compliance)
        self.frame_count += 1
        
        # Create semi-transparent overlay for dashboard
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-350, 10), (w-10, 400), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 30
        
        # Title
        cv2.putText(frame, "📊 COMPLIANCE DASHBOARD", (w-340, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Current Compliance Score
        compliance_color = self.get_compliance_color(current_compliance)
        cv2.putText(frame, f"Current Compliance:", (w-340, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        
        # Draw compliance bar
        bar_x, bar_y = w-340, y_offset
        bar_w, bar_h = 200, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * current_compliance / 100)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), compliance_color, -1)
        cv2.putText(frame, f"{current_compliance:.1f}%", (bar_x+bar_w+10, bar_y+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, compliance_color, 2)
        y_offset += 30
        
        # People Statistics
        cv2.putText(frame, f"People Present: {tracker_stats.get('currently_present', 0)}", 
                   (w-340, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"Total Tracked: {tracker_stats.get('total_people_seen', 0)}", 
                   (w-340, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        # Average Compliance
        avg_compliance = tracker_stats.get('avg_compliance', 0)
        avg_color = self.get_compliance_color(avg_compliance)
        cv2.putText(frame, f"Average Compliance:", (w-340, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(frame, f"{avg_compliance:.1f}%", (w-340, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, avg_color, 2)
        y_offset += 25
        
        # Mask Distribution
        cv2.putText(frame, "Mask Distribution:", (w-340, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        
        dist = tracker_stats.get('mask_distribution', {})
        for mask_type, count in dist.items():
            if count > 0:
                color = self.get_mask_color(mask_type)
                cv2.putText(frame, f"  {mask_type}: {count}", (w-340, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 15
        
        # Compliance trend (mini graph)
        if len(self.compliance_history) > 1:
            graph_y = y_offset + 20
            graph_x = w-340
            graph_w, graph_h = 200, 40
            
            # Draw graph background
            cv2.rectangle(frame, (graph_x, graph_y), (graph_x+graph_w, graph_y+graph_h), 
                         (50, 50, 50), -1)
            
            # Draw trend line
            points = []
            for i, value in enumerate(list(self.compliance_history)[-50:]):
                x = graph_x + int((i / 50) * graph_w)
                y = graph_y + graph_h - int((value / 100) * graph_h)
                points.append((x, y))
            
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 255, 0), 1)
            
            cv2.putText(frame, "Trend", (graph_x, graph_y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Running time
        running_time = int(time.time() - self.start_time)
        hours = running_time // 3600
        minutes = (running_time % 3600) // 60
        seconds = running_time % 60
        time_str = f"⏱️ {hours:02d}:{minutes:02d}:{seconds:02d}"
        cv2.putText(frame, time_str, (w-150, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def get_mask_color(self, mask_type):
        """Get color for mask type in dashboard"""
        colors = {
            'surgical': (255, 0, 0),
            'n95': (0, 255, 0),
            'cloth': (0, 165, 255),
            'proper': (0, 255, 0),
            'below_nose': (0, 165, 255),
            'improper': (0, 0, 255)
        }
        return colors.get(mask_type, (255, 255, 255))
    
    def draw_guide(self, frame):
        """Draw a guide showing proper mask wearing"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h-150), (250, h-10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw guide text
        y_pos = h-130
        cv2.putText(frame, "✅ PROPER MASK:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += 20
        cv2.putText(frame, "  • Cover nose", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 15
        cv2.putText(frame, "  • Cover mouth", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_pos += 20
        cv2.putText(frame, "⚠️ IMPROPER:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        y_pos += 20
        cv2.putText(frame, "  • Nose exposed", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 15
        cv2.putText(frame, "  • Mask below nose", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame