from ultralytics import YOLO
import cv2

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_plate(self, frame, margin=40):
        """
        Detects license plates in a frame.
        Returns: (cropped_plate, (x1, y1, x2, y2)) or (None, None)
        """
        results = self.model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = frame.shape[:2]
                x1m = max(x1 - margin, 0)
                y1m = max(y1 - margin, 0)
                x2m = min(x2 + margin, w)
                y2m = min(y2 + margin, h)
                cropped = frame[y1m:y2m, x1m:x2m]
                return cropped, (x1, y1, x2, y2)
        return None, None
