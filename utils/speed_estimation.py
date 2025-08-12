"""
Speed estimation module with smoothing capabilities.
Place this file in: mlp/OCR-Project/utils/speed_estimator.py
"""
import time
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

class SpeedEstimator:
    """Estimates vehicle speeds with temporal smoothing."""
    
    def __init__(self, 
                 pixels_per_meter: float = 20.0,
                 update_interval: float = 0.5,
                 smoothing_window: int = 5):
        """
        Initialize speed estimator.
        
        Args:
            pixels_per_meter: Calibration factor for pixel to meter conversion
            update_interval: Time interval between speed updates (seconds)
            smoothing_window: Number of speed values to average for smoothing
        """
        self.pixels_per_meter = pixels_per_meter
        self.update_interval = update_interval
        self.smoothing_window = smoothing_window
        
        # Track data storage
        self.track_positions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self.track_speeds: Dict[int, deque] = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.last_speed_update: Dict[int, float] = {}
        self.current_speeds: Dict[int, float] = {}
        
    def update_track_position(self, track_id: int, bbox: List[float], timestamp: float) -> Optional[float]:
        """
        Update position for a track and calculate speed if interval has passed.
        
        Args:
            track_id: Unique track identifier
            bbox: Bounding box [x1, y1, x2, y2]
            timestamp: Current timestamp
            
        Returns:
            Smoothed speed in km/h if updated, None otherwise
        """
        # Calculate center point of bounding box
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Store position with timestamp
        self.track_positions[track_id].append({
            'position': (center_x, center_y),
            'timestamp': timestamp
        })
        
        # Check if we should update speed
        should_update = (
            track_id not in self.last_speed_update or 
            timestamp - self.last_speed_update[track_id] >= self.update_interval
        )
        
        if should_update and len(self.track_positions[track_id]) >= 2:
            speed = self._calculate_speed(track_id)
            if speed is not None:
                # Add to smoothing buffer
                self.track_speeds[track_id].append(speed)
                
                # Calculate smoothed speed
                smoothed_speed = np.mean(list(self.track_speeds[track_id]))
                self.current_speeds[track_id] = smoothed_speed
                self.last_speed_update[track_id] = timestamp
                
                return smoothed_speed
        
        return self.current_speeds.get(track_id)
    
    def _calculate_speed(self, track_id: int) -> Optional[float]:
        """Calculate instantaneous speed for a track."""
        positions = self.track_positions[track_id]
        if len(positions) < 2:
            return None
        
        # Get last two positions
        current = positions[-1]
        previous = positions[-2]
        
        # Calculate distance and time
        dx = current['position'][0] - previous['position'][0]
        dy = current['position'][1] - previous['position'][1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        distance_meters = distance_pixels / self.pixels_per_meter
        
        time_elapsed = current['timestamp'] - previous['timestamp']
        if time_elapsed <= 0:
            return None
        
        # Speed in m/s, convert to km/h
        speed_mps = distance_meters / time_elapsed
        speed_kmph = speed_mps * 3.6
        
        return speed_kmph
    
    def get_current_speed(self, track_id: int) -> Optional[float]:
        """Get current smoothed speed for a track."""
        return self.current_speeds.get(track_id)
    
    def cleanup_old_tracks(self, active_track_ids: set):
        """Remove data for tracks that are no longer active."""
        all_track_ids = set(self.track_positions.keys())
        inactive_tracks = all_track_ids - active_track_ids
        
        for track_id in inactive_tracks:
            if track_id in self.track_positions:
                del self.track_positions[track_id]
            if track_id in self.track_speeds:
                del self.track_speeds[track_id]
            if track_id in self.last_speed_update:
                del self.last_speed_update[track_id]
            if track_id in self.current_speeds:
                del self.current_speeds[track_id]
