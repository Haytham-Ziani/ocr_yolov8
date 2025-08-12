import cv2
import numpy as np
from ultralytics import YOLO

class CarDetector:
    def __init__(self, model_path, confidence_threshold):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """
        Detect cars in the given frame.

        Args:
            frame (np.ndarray): BGR image.

        Returns:
            List[dict]: Each dict contains:
                - 'bbox': (x1, y1, x2, y2)
                - 'confidence': float
        """
        results = self.model.predict(source=frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    'bbox': tuple(xyxy),
                    'confidence': conf
                })

        return detections
