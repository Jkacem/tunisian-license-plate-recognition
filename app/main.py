import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Patch OpenCV for ultralytics compatibility
import cv2
if not hasattr(cv2, 'imshow'):
    def dummy_imshow(winname, mat):
        pass  # Do nothing for imshow
    cv2.imshow = dummy_imshow

if not hasattr(cv2, 'destroyAllWindows'):
    def dummy_destroyAllWindows():
        pass  # Do nothing for destroyAllWindows
    cv2.destroyAllWindows = dummy_destroyAllWindows

if not hasattr(cv2, 'waitKey'):
    def dummy_waitKey(delay=0):
        return -1  # Return -1 to indicate no key pressed
    cv2.waitKey = dummy_waitKey

import cv2
import numpy as np
from detection.detector import PlateDetector
from ocr.ocr_reader import PlateOCR
from format_tun_plate import classify_plate, get_bg_color, is_libyan_plate, format_libyan_plate_from_digits

from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import re
import threading
import time
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta

def contains_arabic(text):
    # Checks if the text contains any Arabic character
    return re.search(r'[\u0600-\u06FF]', text) is not None

def contains_linked_arabic(text):
    # Checks if the text contains two or more linked Arabic characters in a row
    return re.search(r'[\u0600-\u06FF]{2,}', text) is not None

def get_plate_display_text(plate_text):
    """
    Convert plate text to display text based on country code.
    Returns Arabic text for country names.
    """
    if plate_text.startswith("TN"):
        return "تونس"
    elif plate_text.startswith("DZ"):
        return "الجزائر"
    elif plate_text.startswith("LY"):
        return "ليبيا"
    else:
        return plate_text

# === PATH TO MODEL ===
MODEL_PATH = r"C:\Users\SBS\Desktop\detects-Tunisian-license-plate-numbers\Backend\Model\best002.pt"
FONT_PATH = r"fonts/Amiri-Regular.ttf"  # Ensure this path is correct

# Flask app
app = Flask(__name__, template_folder='../templates', static_folder='../static')
PHOTO_PATH = os.path.join(os.path.dirname(__file__), '..', 'test_photo.jpg')

def run_detection_on_photo(photo_path):
    from backend.db import init_db, SessionLocal, PlateRecord
    init_db()
    detector = PlateDetector(MODEL_PATH)
    ocr = PlateOCR()
    frame = cv2.imread(photo_path)
    if frame is None:
        return {'error': 'Photo not found or invalid.'}

    detected_plates = []
    db = SessionLocal()
    try:
        plate_img, bbox = detector.detect_plate(frame)
        if plate_img is not None:
            plate_text = ocr.read_plate_text(plate_img)
            if plate_text and plate_text != "UNKNOWN":
                # Determine display text and DB text based on country code
                if plate_text.startswith("TN"):
                    display_text = "تونس"
                    db_text = plate_text
                elif plate_text.startswith("DZ"):
                    display_text = "الجزائر"
                    db_text = plate_text
                elif plate_text.startswith("LY"):
                    display_text = "ليبيا"
                    db_text = plate_text
                elif contains_arabic(plate_text):
                    db_text = "TN"
                else:
                    db_text = plate_text
                detected_plates.append(db_text)
                exists = db.query(PlateRecord).filter_by(plate_text=db_text).first()
                if not exists:
                    record = PlateRecord(plate_text=db_text)
                    db.add(record)
                    db.commit()
    finally:
        db.close()
    return {'plates': detected_plates}

def draw_arabic_text(frame, text, position, font_path=FONT_PATH, font_size=32, color=(255, 0, 0)):
    """
    Draw Arabic text on an OpenCV frame using PIL.
    """
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    # Convert OpenCV image to PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # Draw the text
    draw.text(position, bidi_text, font=font, fill=color)

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Global variables for detection

detection_running = False
last_detected_plate = None
detection_stats = {
    'total_detections': 0,
    'last_detection_time': None,
    'current_status': 'Stopped'
}

# Plate confirmation buffer
from collections import deque, Counter
PLATE_BUFFER_SIZE = 5
PLATE_CONFIRM_THRESHOLD = 3
recent_plates = deque(maxlen=PLATE_BUFFER_SIZE)

def run_detection():
    """Run the license plate detection in a separate thread"""
    global detection_running, last_detected_plate, detection_stats
    
    # DB imports
    from backend.db import init_db, SessionLocal, PlateRecord
    
    # Initialize DB (create tables if not exist)
    init_db()
    
    # Load components
    detector = PlateDetector(MODEL_PATH)
    ocr = PlateOCR()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        detection_stats['current_status'] = 'Camera Error'
        return

    print("✅ Camera started. Detection running in background.")
    detection_stats['current_status'] = 'Running'

    db = SessionLocal()
    last_saved_plate = None
    last_saved_time = None
    duplicate_timeout = 5  # seconds - minimum time between same plate detections

    try:
        while detection_running:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame.")
                break

            # Detect plate
            plate_img, bbox = detector.detect_plate(frame)
            detected_plate_text = None

            if plate_img is not None:
                detected_plate_text = ocr.read_plate_text(plate_img)
                bg_color = get_bg_color(plate_img)
                
                # Extract all digits from OCR result list for Libyan plates
                digits = []
                if isinstance(detected_plate_text, list):
                    for i, text in enumerate(detected_plate_text):
                        # Extract all digits from each text element while preserving order
                        text_digits = re.findall(r'\d', str(text))
                        digits.extend(text_digits)
                else:
                    digits = re.findall(r'\d', str(detected_plate_text))
                if is_libyan_plate(bg_color):
                    classified_plate = format_libyan_plate_from_digits(digits)
                else:
                    # Handle non-Libyan plates (Tunisian/Algerian)
                    if isinstance(detected_plate_text, list):
                        # Check for Tunisian hints
                        joined_upper = " ".join(detected_plate_text).upper()
                        joined_raw = " ".join(detected_plate_text)
                        
                        if "TN" in joined_upper or re.search(r"تونس", joined_raw):
                            # Import the formatting function
                            from format_tun_plate import format_tunisian_plate_cam_center
                            classified_plate = format_tunisian_plate_cam_center(detected_plate_text)
                        else:
                            # Try Algerian formatting
                            from format_tun_plate import format_algerian_plate
                            dz_formatted = format_algerian_plate(detected_plate_text)
                            if dz_formatted != "UNKNOWN":
                                classified_plate = f"DZ {dz_formatted}"
                            else:
                                # Fallback to Tunisian
                                from format_tun_plate import format_tunisian_plate_cam_center
                                classified_plate = format_tunisian_plate_cam_center(detected_plate_text)
                    else:
                        classified_plate = detected_plate_text
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if classified_plate and classified_plate != "UNKNOWN":
                    if classified_plate.startswith("TN"):
                        display_text = "تونس"
                        db_text = classified_plate
                    elif classified_plate.startswith("DZ"):
                        display_text = "الجزائر"
                        db_text = classified_plate
                    elif classified_plate.startswith("LYB") or classified_plate.endswith("LYB"):
                        display_text = "ليبيا"
                        db_text = classified_plate
                    elif contains_arabic(classified_plate):
                        display_text = "تونس"
                        db_text = "TN"
                    else:
                        display_text = classified_plate
                        db_text = classified_plate
                    print(f"✅ Detected formatted plate: {classified_plate} | Display: {display_text} | To DB: {db_text}")

                    last_detected_plate = db_text
                    detection_stats['total_detections'] += 1
                    detection_stats['last_detection_time'] = datetime.now()

                    try:
                        frame = draw_arabic_text(frame, display_text, (x1, y1 - 40))
                    except Exception as e:
                        print(f"Warning: Could not draw text on frame: {e}")

                    recent_plates.append(db_text)
                    most_common_plate, count = Counter(recent_plates).most_common(1)[0]
                    current_time = datetime.now()
                    exists = db.query(PlateRecord).filter_by(plate_text=most_common_plate).first()
                    is_same_as_last = (last_saved_plate == most_common_plate)
                    is_within_timeout = (last_saved_time and (current_time - last_saved_time).total_seconds() < duplicate_timeout)
                    if count >= PLATE_CONFIRM_THRESHOLD and not exists and not (is_same_as_last and is_within_timeout):
                        try:
                            record = PlateRecord(plate_text=most_common_plate, timestamp=current_time)
                            db.add(record)
                            db.commit()
                            last_saved_plate = most_common_plate
                            last_saved_time = current_time
                            print(f"✅ Saved to DB: {most_common_plate}")
                        except Exception as e:
                            db.rollback()
                            print(f"DB Error: {e}")
                    else:
                        if exists:
                            print(f"⏭️ Plate already exists in DB: {most_common_plate}")
                        elif count < PLATE_CONFIRM_THRESHOLD:
                            print(f"⏭️ Plate not confirmed yet: {most_common_plate} ({count}/{PLATE_BUFFER_SIZE})")
                        else:
                            print(f"⏭️ Same plate detected recently, skipping: {most_common_plate}")
                elif detected_plate_text == "UNKNOWN":
                    print(f"❓ Plate detected but format unknown - raw OCR may have failed")

            # Don't show live feed window to avoid GUI issues
            # cv2.imshow("YOLOv8 License Plate Detection", frame)

            # Check for quit condition (use a different method)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        db.close()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass  # Ignore GUI errors
        detection_stats['current_status'] = 'Stopped'

# Flask routes
@app.route('/')
def dashboard():
    """Serve the dashboard page"""
    from backend.db import SessionLocal, PlateRecord, CarInfo

    db = SessionLocal()
    try:
        # Get all plates, ordered by most recent first
        plates = db.query(PlateRecord).order_by(PlateRecord.timestamp.desc()).all()
        # For each plate, get car class if exists
        plate_data = []
        for plate in plates:
            car = db.query(CarInfo).filter_by(plate_text=plate.plate_text).first()
            car_class = car.car_class if car else "Unknown"
            plate_data.append({
                'id': plate.id,
                'plate_text': plate.plate_text,
                'car_class': car_class,
                'timestamp': plate.timestamp
            })
        return render_template('dashboard.html', plates=plate_data)
    finally:
        db.close()

@app.route('/api/plates')
def api_plates():
    """API endpoint to get all plates"""
    from backend.db import SessionLocal, PlateRecord
    
    db = SessionLocal()
    try:
        plates = db.query(PlateRecord).order_by(PlateRecord.timestamp.desc()).all()
        return jsonify([{
            'id': plate.id,
            'plate_text': plate.plate_text,
            'timestamp': plate.timestamp.isoformat()
        } for plate in plates])
    finally:
        db.close()

@app.route('/api/stats')
def api_stats():
    """API endpoint to get detection statistics"""
    return jsonify(detection_stats)

# New workflow: capture photo and detect
@app.route('/api/capture_photo', methods=['POST'])
def api_capture_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(PHOTO_PATH, frame)
        return jsonify({'status': 'photo_saved'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to capture photo'})

@app.route('/api/detect_photo', methods=['POST'])
def api_detect_photo():
    result = run_detection_on_photo(PHOTO_PATH)
    return jsonify(result)

@app.route('/api/start', methods=['POST'])
def start_detection():
    """API endpoint to start detection"""
    global detection_running, detection_stats
    
    if not detection_running:
        detection_running = True
        detection_stats['current_status'] = 'Starting...'
        
        # Start detection in a separate thread
        detection_thread = threading.Thread(target=run_detection)
        detection_thread.daemon = True
        detection_thread.start()
        
        return jsonify({'status': 'success', 'message': 'Detection started'})
    else:
        return jsonify({'status': 'error', 'message': 'Detection already running'})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """API endpoint to stop detection"""
    global detection_running, detection_stats
    
    detection_running = False
    detection_stats['current_status'] = 'Stopping...'
    
    return jsonify({'status': 'success', 'message': 'Detection stopped'})

def main():
    """Main function to run the Flask app"""
    print("🚀 Starting License Plate Detection System")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔌 API endpoints:")
    print("   - GET  /api/plates - Get all plates")
    print("   - GET  /api/stats - Get detection stats")
    print("   - POST /api/start - Start detection")
    print("   - POST /api/stop - Stop detection")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()