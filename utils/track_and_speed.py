import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from utils.bytetrack.speed_estimator import SpeedEstimator


class TrackAndSpeed:
    def __init__(self, frame_rate, pixel_to_meter, confidence_threshold=0.3):
        # ByteTrack args setup
        class TrackingArgs:
            def __init__(self):
                self.track_thresh = confidence_threshold
                self.track_buffer = 30
                self.match_thresh = 0.8
                self.aspect_ratio_thresh = 1.6
                self.min_box_area = 1.0
                self.mot20 = False

        self.tracker = BYTETracker(TrackingArgs(), frame_rate=frame_rate)

        self.speed_estimator = SpeedEstimator(
            pixels_per_meter=1.0 / pixel_to_meter,
            update_interval=0.3,  # every 300ms
            smoothing_window=5
        )

        self.frame_rate = frame_rate
        self.current_time = 0.0

    def update(self, detections, frame):
        """
        Track cars and estimate speed.

        Args:
            detections (List[dict]): [{'bbox': (x1,y1,x2,y2), 'confidence': float}]
            frame (np.ndarray): Current video frame.

        Returns:
            List[dict]: [{
                'track_id': int,
                'bbox': (x1, y1, x2, y2),
                'speed': float,
                'confidence': float
            }]
        """
        self.current_time += 1.0 / self.frame_rate

        # Prepare detections for tracker: [x1, y1, x2, y2, score]
        if len(detections) > 0:
            dets_np = np.array([list(det['bbox']) + [det['confidence']] for det in detections])
        else:
            dets_np = np.empty((0, 5))

        online_targets = self.tracker.update(dets_np, frame.shape[:2], frame.shape[:2])
        active_track_ids = set()
        output = []

        for target in online_targets:
            tid = int(target.track_id)
            x1, y1, x2, y2 = [int(i) for i in target.tlbr]
            bbox = (x1, y1, x2, y2)

            current_speed = self.speed_estimator.update_track_position(tid, bbox, self.current_time)
            active_track_ids.add(tid)

            output.append({
                'track_id': tid,
                'bbox': bbox,
                'speed': float(current_speed) if current_speed is not None else 0.0,
                'confidence': float(target.score)
            })

        self.speed_estimator.cleanup_old_tracks(active_track_ids)
        return output
