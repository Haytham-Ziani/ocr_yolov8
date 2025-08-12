import cv2
import numpy as np
from ultralytics import YOLO

class CharacterSegmentor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        # Optional: Map YOLO class IDs to characters
        self.id2char = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'أ', 11: 'ب', 12: 'و', 13: 'د', 14: 'ھ', 15: 'W'
        }

    def extract_text(self, plate_crop):
        """
        Detects characters in a license plate image and returns them as a string.

        Args:
            plate_crop (np.ndarray): BGR image of license plate.

        Returns:
            str: Recognized license plate string.
        """
        results = self.model.predict(source=plate_crop, conf=0.3, verbose=False)

        characters = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                center_x = (x1 + x2) // 2

                characters.append({
                    'char': self.id2char.get(cls_id, '?'),
                    'center_x': center_x
                })

        # Sort characters from left to right
        sorted_chars = sorted(characters, key=lambda c: c['center_x'])

        # Concatenate all characters into a string
        license_text = ''.join([c['char'] for c in sorted_chars])
        return license_text
