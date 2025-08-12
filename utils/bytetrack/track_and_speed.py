import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from utils.bytetrack.speed_estimator import SpeedEstimator
from collections import defaultdict, deque
import cv2


class TrackAndSpeed:
    def __init__(self, fps, pixel_to_meter, speed_limit_kmph=None, confidence_threshold=0.3):
        # Improved ByteTrack args setup
        class TrackingArgs:
            def __init__(self):
                self.track_thresh = confidence_threshold
                self.track_buffer = 60  # Increased from 30 to keep tracks alive longer
                self.match_thresh = 0.7  # Lowered from 0.8 for more lenient matching
                self.aspect_ratio_thresh = 3.0  # Increased for more flexibility
                self.min_box_area = 100  # Increased to filter out noise
                self.mot20 = False

        self.tracker = BYTETracker(TrackingArgs())

        self.speed_estimator = SpeedEstimator(
            pixels_per_meter=1.0 / pixel_to_meter,
            update_interval=0.2,  # Reduced from 0.3 for more frequent updates
            smoothing_window=7  # Increased for better smoothing
        )

        self.fps = fps
        self.speed_limit_kmph = speed_limit_kmph
        self.current_time = 0.0

        # Track association and ID stability
        self.track_history = defaultdict(deque)  # Store recent positions
        self.lost_tracks = {}  # Recently lost tracks with their last positions
        self.track_associations = {}  # Map new IDs to original IDs
        self.next_stable_id = 1
        self.stable_id_map = {}  # ByteTrack ID -> Our stable ID

        # Speed calculation safeguards
        self.speed_calculations = set()  # Track which speeds we've calculated
        self.violation_records = set()  # Prevent duplicate violation saves

        # Track quality metrics
        self.track_confidence = defaultdict(list)
        self.track_ages = defaultdict(int)

        # Parameters for track association
        self.max_association_distance = 100  # pixels
        self.max_frames_gap = 15  # frames
        self.min_track_age = 5  # minimum frames before considering track stable

    def _calculate_bbox_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _find_matching_lost_track(self, bbox, current_frame):
        """Find if a new track matches a recently lost track"""
        current_center = self._calculate_bbox_center(bbox)

        best_match = None
        best_distance = float('inf')

        for lost_id, (last_pos, last_frame, original_stable_id) in list(self.lost_tracks.items()):
            frame_gap = current_frame - last_frame

            if frame_gap <= self.max_frames_gap:
                distance = self._calculate_distance(current_center, last_pos)

                if distance < self.max_association_distance and distance < best_distance:
                    best_match = (lost_id, original_stable_id)
                    best_distance = distance

        return best_match

    def _get_stable_id(self, bytetrack_id, bbox, current_frame):
        """Get or create a stable ID for the given ByteTrack ID"""
        if bytetrack_id in self.stable_id_map:
            return self.stable_id_map[bytetrack_id]

        # Check if this might be a re-appeared track
        match = self._find_matching_lost_track(bbox, current_frame)

        if match:
            lost_id, original_stable_id = match
            # Remove from lost tracks and reassign
            del self.lost_tracks[lost_id]
            self.stable_id_map[bytetrack_id] = original_stable_id
            return original_stable_id
        else:
            # Create new stable ID
            stable_id = self.next_stable_id
            self.next_stable_id += 1
            self.stable_id_map[bytetrack_id] = stable_id
            return stable_id

    def _update_track_quality(self, stable_id, confidence, bbox):
        """Update track quality metrics"""
        self.track_confidence[stable_id].append(confidence)
        self.track_ages[stable_id] += 1

        # Keep only recent confidence values
        if len(self.track_confidence[stable_id]) > 10:
            self.track_confidence[stable_id].pop(0)

    def _is_track_stable(self, stable_id):
        """Check if track is stable enough for speed calculation"""
        if stable_id not in self.track_ages:
            return False

        age = self.track_ages[stable_id]
        if age < self.min_track_age:
            return False

        # Check confidence consistency
        confidences = self.track_confidence[stable_id]
        if len(confidences) >= 3:
            avg_confidence = np.mean(confidences)
            return avg_confidence > 0.5

        return True

    def _should_calculate_speed(self, stable_id, current_frame):
        """Determine if we should calculate speed for this track"""
        # Create unique identifier for this speed calculation window
        calculation_window = current_frame // 10  # Group by 10-frame windows
        calc_id = f"{stable_id}_{calculation_window}"

        if calc_id in self.speed_calculations:
            return False

        self.speed_calculations.add(calc_id)
        return True

    def _should_record_violation(self, stable_id, speed, current_frame):
        """Check if we should record a speed violation"""
        if self.speed_limit_kmph is None or speed <= self.speed_limit_kmph:
            return False

        # Create violation window to prevent duplicates
        violation_window = current_frame // 30  # Group by 30-frame windows (1 second at 30fps)
        violation_id = f"{stable_id}_{violation_window}"

        if violation_id in self.violation_records:
            return False

        self.violation_records.add(violation_id)
        return True

    def update(self, detections, frame):
        """
        Track cars and estimate speed with improved ID stability.

        Args:
            detections (List[dict]): [{'bbox': (x1,y1,x2,y2), 'confidence': float}]
            frame (np.ndarray): Current video frame.

        Returns:
            List[dict]: [{
                'track_id': int,  # Stable ID
                'bbox': (x1, y1, x2, y2),
                'speed': float,
                'confidence': float,
                'is_violation': bool,
                'track_age': int
            }]
        """
        current_frame = int(self.current_time * self.fps)
        self.current_time += 1.0 / self.fps

        # Prepare detections for tracker: [x1, y1, x2, y2, score]
        if len(detections) > 0:
            dets_np = np.array([list(det['bbox']) + [det['confidence']] for det in detections])
        else:
            dets_np = np.empty((0, 5))

        online_targets = self.tracker.update(dets_np, frame.shape[:2], frame.shape[:2])

        # Track which ByteTrack IDs are still active
        current_bytetrack_ids = set()
        active_stable_ids = set()
        output = []

        for target in online_targets:
            bytetrack_id = int(target.track_id)
            x1, y1, x2, y2 = [int(i) for i in target.tlbr]
            bbox = (x1, y1, x2, y2)
            confidence = float(target.score)

            # Get stable ID
            stable_id = self._get_stable_id(bytetrack_id, bbox, current_frame)

            current_bytetrack_ids.add(bytetrack_id)
            active_stable_ids.add(stable_id)

            # Update track quality metrics
            self._update_track_quality(stable_id, confidence, bbox)

            # Update position history
            center = self._calculate_bbox_center(bbox)
            self.track_history[stable_id].append((center, current_frame))
            if len(self.track_history[stable_id]) > 20:  # Keep last 20 positions
                self.track_history[stable_id].popleft()

            # Calculate speed only for stable tracks
            current_speed = 0.0
            is_violation = False

            if self._is_track_stable(stable_id):
                if self._should_calculate_speed(stable_id, current_frame):
                    speed = self.speed_estimator.update_track_position(
                        stable_id, bbox, self.current_time
                    )
                    if speed is not None:
                        current_speed = float(speed)
                        is_violation = self._should_record_violation(
                            stable_id, current_speed, current_frame
                        )

            output.append({
                'track_id': stable_id,
                'bbox': bbox,
                'speed': current_speed,
                'confidence': confidence,
                'is_violation': is_violation,
                'track_age': self.track_ages[stable_id],
                'is_stable': self._is_track_stable(stable_id)
            })

        # Handle lost tracks
        lost_bytetrack_ids = set(self.stable_id_map.keys()) - current_bytetrack_ids

        for lost_id in lost_bytetrack_ids:
            stable_id = self.stable_id_map[lost_id]

            # Store last known position for potential re-association
            if stable_id in self.track_history and len(self.track_history[stable_id]) > 0:
                last_pos, last_frame = self.track_history[stable_id][-1]
                self.lost_tracks[lost_id] = (last_pos, current_frame, stable_id)

            # Clean up
            del self.stable_id_map[lost_id]

        # Clean up old lost tracks
        self.lost_tracks = {
            k: v for k, v in self.lost_tracks.items()
            if current_frame - v[1] <= self.max_frames_gap
        }

        # Clean up speed estimator
        self.speed_estimator.cleanup_old_tracks(active_stable_ids)

        # Clean up old calculation records
        old_calculations = {
            calc_id for calc_id in self.speed_calculations
            if int(calc_id.split('_')[-1]) < (current_frame // 10) - 10
        }
        self.speed_calculations -= old_calculations

        # Clean up old violation records
        old_violations = {
            viol_id for viol_id in self.violation_records
            if int(viol_id.split('_')[-1]) < (current_frame // 30) - 30
        }
        self.violation_records -= old_violations

        return output

    def get_track_statistics(self):
        """Get statistics about track stability"""
        return {
            'total_tracks': len(self.track_ages),
            'active_tracks': len([age for age in self.track_ages.values() if age > 0]),
            'stable_tracks': len([sid for sid in self.track_ages.keys() if self._is_track_stable(sid)]),
            'lost_tracks_pending': len(self.lost_tracks),
            'speed_calculations_made': len(self.speed_calculations),
            'violations_recorded': len(self.violation_records)
        }