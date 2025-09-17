import easyocr
import cv2
import numpy as np
import re


# Characters to keep: all Arabic letters and numbers
ALLOWED_ARABIC = set(
    "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوي"  # Arabic alphabet
    "ىةﻻﻷﻹﻵ"  # Common ligatures and forms
    "تونس"    # Keep your special case too
)
ALLOWED_NUMBERS = set("0123456789")


class PlateOCR:
    def __init__(self):
        # Use English EasyOCR for better digit recognition (like test.py)
        self.reader = easyocr.Reader(['en'], gpu=False)


    def preprocess_plate(self, plate_img):
        """
        Resize, sharpen, and upscale the plate image for better OCR accuracy.
        """
        target_size = (640, 160)
        plate_resized = cv2.resize(plate_img, target_size, interpolation=cv2.INTER_LANCZOS4)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        plate_sharp = cv2.filter2D(plate_resized, -1, kernel)
        plate_upscaled = cv2.resize(plate_sharp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return plate_upscaled

    def read_plate_text(self, plate_img):
        """
        Reads text from a plate image using English EasyOCR.
        Returns: raw OCR results (list) or "UNKNOWN"
        """
        # Convert to grayscale for better OCR (like test.py)
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Use English EasyOCR for better digit recognition (like test.py)
        ocr_results = self.reader.readtext(gray_plate, detail=0, paragraph=False)
        
        if ocr_results:
            # Return raw OCR results for proper classification in main application
            return ocr_results
        
        return "UNKNOWN"

    def read_plate_text_with_position(self, plate_img, camera_position='center'):
        """
        Reads text from a plate image and returns joined OCR result or "UNKNOWN".
        Args:
            plate_img: The plate image
            camera_position: 'center', 'left', or 'right' (unused)
        Returns: joined OCR result (str) or "UNKNOWN"
        """
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        ocr_results = self.reader.readtext(gray_plate, detail=0, paragraph=False)
        if ocr_results:
            return " ".join(ocr_results)
        return "UNKNOWN"