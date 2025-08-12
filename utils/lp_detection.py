import cv2
import numpy as np
from ultralytics import YOLO

class LicensePlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, car_crop, confidence_threshold=0.3):
        """
        Detect license plates in a cropped image of a car.

        Args:
            car_crop (np.ndarray): BGR image of the car.

        Returns:
            List[dict]: Each dict contains:
                - 'bbox': (x1, y1, x2, y2)
                - 'confidence': float
        """
        results = self.model.predict(source=car_crop, conf=confidence_threshold, verbose=False)  # you can change threshold if needed

        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    'bbox': tuple(xyxy),
                    'confidence': conf
                })

        return detections
