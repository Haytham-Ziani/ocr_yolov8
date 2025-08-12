# main.py - Speed Camera Pipeline
import cv2
import numpy as np
from datetime import datetime
import os
import json
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import uuid
import argparse

# Import your project modules
from utils.car_detection import CarDetector
from utils.lp_detection import LicensePlateDetector
from utils.char_segmentation import CharacterSegmentor
from utils.bytetrack.track_and_speed import TrackAndSpeed
from utils.speed_estimation import SpeedEstimator


@dataclass
class Detection:
    """Data structure for car detection"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    track_id: int
    speed: float
    timestamp: datetime


@dataclass
class ViolationRecord:
    """Data structure for speed violations"""
    detection: Detection
    license_plate: str
    plate_confidence: float
    image_path: str
    violation_id: str


class SpeedCameraPipeline:
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        """Initialize the speed camera pipeline with configuration"""

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize models
        self.car_detector = CarDetector(
            model_path=self.config['models']['car_detector_path'],
            confidence_threshold=self.config['detection']['car_confidence_threshold']
        )

        self.lp_detector = LicensePlateDetector(
            model_path=self.config['models']['lp_detector_path'],
        )

        self.char_segmentor = CharacterSegmentor(
            model_path=self.config['models']['char_segmentor_path']
        )

        # Initialize tracking and speed estimation
        self.tracker_speed = TrackAndSpeed(
            pixel_to_meter=self.config['speed']['pixel_to_meter'],
            fps=self.config['speed']['fps']
        )

        # Configuration parameters
        self.speed_threshold = self.config['speed']['speed_threshold']
        self.output_dir = self.config['output']['results_dir']

        # Create output directories
        self._setup_output_directories()

        # Internal state
        self.violation_records = []
        self.processed_tracks = set()  # To avoid duplicate violations for same track
        self.frame_count = 0

        print(f"Pipeline initialized with speed threshold: {self.speed_threshold} km/h")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Default configuration if config file is not found"""
        return {
            'models': {
                'car_detector_path': 'models/car_detector.pt',
                'lp_detector_path': 'models/lp_detector.pt',
                'char_segmentor_path': 'models/char_segmentor.pt'
            },
            'detection': {
                'car_confidence_threshold': 0.5,
                'lp_confidence_threshold': 0.5
            },
            'speed': {
                'speed_threshold': 100.0,  # km/h
                'pixel_to_meter': 0.0001,  # Adjust based on your camera setup
                'fps': 30
            },
            'output': {
                'results_dir': 'results',
                'save_violations': True,
                'save_video': True
            }
        }

    def _setup_output_directories(self):
        """Create necessary output directories"""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, "annotated_frames"),
            os.path.join(self.output_dir, "logs"),
            "data/tracking_logs"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def process_frame(self, frame: np.ndarray, frame_time: float = None) -> Tuple[List[ViolationRecord], np.ndarray]:
        """
        Process a single frame through the entire pipeline
        Args:
            frame: Input frame
            frame_time: Video timestamp in seconds (if None, uses frame count / fps)
        Returns:
            Tuple of (new violations list, annotated frame)
        """
        self.frame_count += 1
        new_violations = []

        # Calculate frame timestamp
        if frame_time is None:
            frame_time = self.frame_count / self.config['speed']['fps']

        # Step 1: Detect cars
        car_detections = self.car_detector.detect(frame)

        if not car_detections:
            tracked_objects = []
            annotated_frame = self._annotate_frame_for_video(frame, tracked_objects, new_violations)
            return new_violations, annotated_frame

        # Step 2: Update tracker and calculate speeds
        tracked_objects = self.tracker_speed.update(car_detections, frame)

        # Step 3: Process each tracked object
        for track in tracked_objects:
            track_id = track['track_id']
            bbox = track['bbox']
            speed = track.get('speed', 0.0)
            confidence = track.get('confidence', 0.0)

            # Step 4: Check if speed exceeds threshold
            if speed > self.speed_threshold and track_id not in self.processed_tracks:
                # Mark this track as processed to avoid duplicates
                self.processed_tracks.add(track_id)

                # Create detection object with video timestamp
                detection = Detection(
                    bbox=bbox,
                    confidence=confidence,
                    track_id=track_id,
                    speed=speed,
                    timestamp=datetime.fromtimestamp(frame_time)
                )

                # Step 5: Process the violation
                violation = self._process_violation(frame, detection)
                if violation:
                    new_violations.append(violation)
                    self.violation_records.append(violation)
                    print(f"VIOLATION DETECTED! Track {track_id}: {speed:.1f} km/h")

        annotated_frame = self._annotate_frame_for_video(frame, tracked_objects, new_violations)
        return new_violations, annotated_frame

    def _process_violation(self, frame: np.ndarray, detection: Detection) -> Optional[ViolationRecord]:
        """
        Process a speed violation: extract plate, run OCR, save annotated image
        """
        try:
            # Extract car region from frame
            x1, y1, x2, y2 = detection.bbox
            car_crop = frame[y1:y2, x1:x2]

            if car_crop.size == 0:
                return None

            # Step 6: Detect license plate in the car crop with confidence threshold
            plate_detections = self.lp_detector.detect(
                car_crop,
                confidence_threshold=self.config['detection']['lp_confidence_threshold']
            )

            if not plate_detections:
                print(f"No license plate found for violation {detection.track_id}")
                return None

            # Take the most confident plate detection
            best_plate = max(plate_detections, key=lambda x: x.get('confidence', 0))

            # Extract plate region
            px1, py1, px2, py2 = best_plate['bbox']
            plate_crop = car_crop[py1:py2, px1:px2]

            if plate_crop.size == 0:
                return None

            # Step 7: Run OCR on license plate
            license_plate_text = self.char_segmentor.extract_text(plate_crop)

            if not license_plate_text or len(license_plate_text.strip()) == 0:
                print(f"Could not extract text from license plate for track {detection.track_id}")
                return None

            # Step 8: Create annotated image and save
            violation_id = str(uuid.uuid4())[:8]  # Short ID for filenames
            annotated_image = self._create_annotated_image(
                frame, detection, best_plate, license_plate_text
            )

            # Save the annotated image
            timestamp_str = detection.timestamp.strftime("%Y%m%d_%H%M%S")
            image_filename = f"violation_{timestamp_str}_{violation_id}.jpg"
            image_path = os.path.join(self.output_dir, "annotated_frames", image_filename)
            cv2.imwrite(image_path, annotated_image)

            # Create violation record
            violation = ViolationRecord(
                detection=detection,
                license_plate=license_plate_text.strip(),
                plate_confidence=best_plate.get('confidence', 0.0),
                image_path=image_path,
                violation_id=violation_id
            )

            # Save violation data as JSON
            self._save_violation_data(violation, timestamp_str)

            return violation

        except Exception as e:
            print(f"Error processing violation for track {detection.track_id}: {e}")
            return None

    def _create_annotated_image(self, frame: np.ndarray, detection: Detection,
                                plate_detection: dict, license_plate: str) -> np.ndarray:
        """
        Create an annotated image with bounding boxes and violation info
        """
        annotated = frame.copy()
        x1, y1, x2, y2 = detection.bbox

        # Draw car bounding box (red for violation)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Draw license plate bounding box (within car region)
        px1, py1, px2, py2 = plate_detection['bbox']
        plate_x1, plate_y1 = x1 + px1, y1 + py1
        plate_x2, plate_y2 = x1 + px2, y1 + py2
        cv2.rectangle(annotated, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 255, 0), 2)

        # Add text annotations with background for visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Prepare text information
        texts = [
            f"SPEED VIOLATION: {detection.speed:.1f} km/h",
            f"License Plate: {license_plate}",
            f"Track ID: {detection.track_id}",
            f"Time: {detection.timestamp.strftime('%H:%M:%S')}"
        ]

        # Calculate text positioning
        text_y_start = max(50, y1 - 120)

        for i, text in enumerate(texts):
            y_pos = text_y_start + (i * 30)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Draw background rectangle for text
            padding = 10
            bg_x1 = x1 - padding
            bg_y1 = y_pos - text_height - padding
            bg_x2 = x1 + text_width + padding
            bg_y2 = y_pos + padding

            cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), 2)

            # Draw text
            cv2.putText(annotated, text, (x1, y_pos), font, font_scale, (255, 255, 255), thickness)

        return annotated

    def _annotate_frame_for_video(self, frame: np.ndarray, tracked_objects: List[dict],
                                  violations: List[ViolationRecord]) -> np.ndarray:
        """
        Create annotated frame for output video with all tracked objects and violations
        """
        annotated = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Create a set of violation track IDs for easy lookup
        violation_track_ids = {v.detection.track_id for v in violations}

        # Draw all tracked objects
        for track in tracked_objects:
            track_id = track['track_id']
            bbox = track['bbox']
            speed = track.get('speed', 0.0)
            confidence = track.get('confidence', 0.0)

            x1, y1, x2, y2 = bbox

            # Color coding: Red for violations, Green for normal traffic
            if track_id in violation_track_ids:
                box_color = (0, 0, 255)  # Red for violation
                text_bg_color = (0, 0, 255)
                label = f"VIOLATION"
            else:
                box_color = (0, 255, 0)  # Green for normal
                text_bg_color = (0, 128, 0)
                label = f"NORMAL"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Prepare text information
            speed_text = f"{speed:.1f} km/h"
            id_text = f"ID:{track_id}"

            # Draw text background and text
            texts = [label, speed_text, id_text]
            text_y_start = max(20, y1 - 60)

            for i, text in enumerate(texts):
                y_pos = text_y_start + (i * 20)
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                # Background rectangle
                cv2.rectangle(annotated, (x1, y_pos - text_height - 3),
                              (x1 + text_width + 6, y_pos + 3), text_bg_color, -1)

                # Text
                cv2.putText(annotated, text, (x1 + 3, y_pos), font, font_scale, (255, 255, 255), thickness)

        # Add license plate information for violations
        for violation in violations:
            x1, y1, x2, y2 = violation.detection.bbox
            plate_text = f"Plate: {violation.license_plate}"

            # Position below the main tracking info
            y_pos = max(20, y1 - 20)
            (text_width, text_height), _ = cv2.getTextSize(plate_text, font, font_scale, thickness)

            # Background rectangle
            cv2.rectangle(annotated, (x1, y_pos - text_height - 3),
                          (x1 + text_width + 6, y_pos + 3), (255, 0, 0), -1)

            # Text
            cv2.putText(annotated, plate_text, (x1 + 3, y_pos), font, font_scale, (255, 255, 255), thickness)

        # Add frame info in top-left corner
        frame_info = f"Frame: {self.frame_count} | Violations: {len(self.violation_records)}"
        cv2.rectangle(annotated, (10, 10), (400, 35), (0, 0, 0), -1)
        cv2.putText(annotated, frame_info, (15, 30), font, 0.5, (255, 255, 255), 1)

        return annotated

    def _save_violation_data(self, violation: ViolationRecord, timestamp_str: str):
        """
        Save violation data as JSON for record keeping
        """
        data = {
            "violation_id": violation.violation_id,
            "timestamp": violation.detection.timestamp.isoformat(),
            "track_id": violation.detection.track_id,
            "speed": round(violation.detection.speed, 2),
            "speed_threshold": self.speed_threshold,
            "license_plate": violation.license_plate,
            "plate_confidence": round(violation.plate_confidence, 3),
            "bbox": violation.detection.bbox,
            "image_path": violation.image_path,
            "frame_number": self.frame_count
        }

        json_filename = f"violation_{timestamp_str}_{violation.violation_id}.json"
        json_path = os.path.join(self.output_dir, "logs", json_filename)

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_violation_summary(self) -> Dict:
        """
        Get summary statistics of violations
        """
        if not self.violation_records:
            return {"total_violations": 0}

        speeds = [v.detection.speed for v in self.violation_records]

        return {
            "total_violations": len(self.violation_records),
            "average_speed": round(np.mean(speeds), 2),
            "max_speed": round(np.max(speeds), 2),
            "min_speed": round(np.min(speeds), 2),
            "unique_plates": len(set(v.license_plate for v in self.violation_records)),
            "frames_processed": self.frame_count
        }

    def cleanup_old_tracks(self, max_age_frames: int = 300):
        """
        Clean up old track IDs to prevent memory issues
        Also cleans tracker and speed estimator internal states
        """
        # Clean processed tracks
        if len(self.processed_tracks) > max_age_frames:
            tracks_list = list(self.processed_tracks)
            self.processed_tracks = set(tracks_list[-max_age_frames // 2:])

        # Clean tracker and speed estimator internal states
        if hasattr(self.tracker_speed, 'cleanup_old_tracks'):
            self.tracker_speed.cleanup_old_tracks(max_age_frames)

        print(f"Cleaned up old tracks. Current processed tracks: {len(self.processed_tracks)}")


def process_video(pipeline: SpeedCameraPipeline, video_path: str, output_video_path: str = None):
    """Process a video file through the pipeline"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

    # Setup video writer if output is requested
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_video_path}")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Calculate frame timestamp in seconds
            frame_time = frame_count / fps

            # Process frame through pipeline
            violations, annotated_frame = pipeline.process_frame(frame, frame_time)

            # Log violations
            for violation in violations:
                print(f"Frame {frame_count} ({frame_time:.2f}s): VIOLATION - Track {violation.detection.track_id}, "
                      f"Speed: {violation.detection.speed:.1f} km/h, Plate: {violation.license_plate}")

            # Write annotated frame to output video
            if out is not None:
                out.write(annotated_frame)

            # Cleanup old tracks periodically
            if frame_count % 100 == 0:
                pipeline.cleanup_old_tracks()
                print(f"Processed {frame_count}/{total_frames} frames... ({(frame_count / total_frames) * 100:.1f}%)")

            # Optional: break for testing
            # if frame_count > 1000:  # Process only first 1000 frames for testing
            #     break

    finally:
        cap.release()
        if out is not None:
            out.release()

        print(f"Video processing complete. Processed {frame_count} frames.")

        # Print summary
        summary = pipeline.get_violation_summary()
        print("\n" + "=" * 50)
        print("VIOLATION SUMMARY:")
        print("=" * 50)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 50)


def main():
    """Main function to run the speed camera pipeline"""
    parser = argparse.ArgumentParser(description='Speed Camera Pipeline')
    parser.add_argument('--config', type=str, default='configs/pipeline_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--input', type=str, default='data/video_input.mp4',
                        help='Input video path')
    parser.add_argument('--output', type=str, default='results/video_output.mp4',
                        help='Output video path (optional)')
    parser.add_argument('--no-video-output', action='store_true',
                        help='Skip video output generation')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = SpeedCameraPipeline(config_path=args.config)

    # Determine output video path
    output_video = None if args.no_video_output else args.output

    # Process video
    process_video(pipeline, args.input, output_video)


if __name__ == "__main__":
    main()