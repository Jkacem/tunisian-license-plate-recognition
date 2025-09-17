import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import logging
import re
from ocr.ocr_reader import PlateOCR
from format_tun_plate import get_bg_color, classify_plate, is_libyan_plate, format_libyan_plate_from_digits

# Load YOLO model
try:
    model = YOLO(r"c:\Users\SBS\Desktop\detects-Tunisian-license-plate-numbers\Backend\Model\best002.pt")
    print("✅ Modèle YOLO chargé avec succès.")
except Exception as e:
    logging.error(f"❌ Erreur de chargement du modèle: {e}")    
    raise e

ocr = PlateOCR()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    print("✅ Camera opened. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        try:
            results = model(frame)
            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    plate_roi = frame[y1:y2, x1:x2]
                    if plate_roi.size != 0:
                        # Let PlateOCR classify TN vs DZ and format
                        ocr_text = ocr.read_plate_text(plate_roi)
                        bg_color = get_bg_color(plate_roi)
                        
                        # Extract all digits from OCR result list for Libyan plates
                        digits = []
                        if isinstance(ocr_text, list):
                            for i, text in enumerate(ocr_text):
                                # Extract all digits from each text element while preserving order
                                text_digits = re.findall(r'\d', str(text))
                                digits.extend(text_digits)
                        else:
                            digits = re.findall(r'\d', str(ocr_text))
                        if is_libyan_plate(bg_color):
                            formatted_plate = format_libyan_plate_from_digits(digits)
                        else:
                            # Handle non-Libyan plates (Tunisian/Algerian)
                            if isinstance(ocr_text, list):
                                # Check for Tunisian hints
                                joined_upper = " ".join(ocr_text).upper()
                                joined_raw = " ".join(ocr_text)
                                
                                if "TN" in joined_upper or re.search(r"تونس", joined_raw):
                                    # Import the formatting function
                                    from format_tun_plate import format_tunisian_plate_cam_center
                                    formatted_plate = format_tunisian_plate_cam_center(ocr_text)
                                else:
                                    # Try Algerian formatting
                                    from format_tun_plate import format_algerian_plate
                                    dz_formatted = format_algerian_plate(ocr_text)
                                    if dz_formatted != "UNKNOWN":
                                        formatted_plate = f"DZ {dz_formatted}"
                                    else:
                                        # Fallback to Tunisian
                                        from format_tun_plate import format_tunisian_plate_cam_center
                                        formatted_plate = format_tunisian_plate_cam_center(ocr_text)
                            else:
                                formatted_plate = ocr_text
                        print(f"✅ Formatted plate: {formatted_plate}")
                        cv2.putText(frame, formatted_plate, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as e:
            print(f"❌ Erreur lors du traitement de la frame: {e}")
        cv2.imshow('YOLO License Plate Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
