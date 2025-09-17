import re
import cv2
import numpy as np
def format_tunisian_plate_cam_center(texts):
    characters = []
    for text in texts:
        for char in text:
            if char.isalnum():
                characters.append(char)

    digits = [c for c in characters if c.isdigit()]
    # print(f" tous les chiffres: {digits}")

    if len(digits) < 3 or len(digits) > 10:
        return "UNKNOWN"

    if len(digits) == 3:
        return f"{digits[0]}{digits[1]} TN {digits[2]}"
    elif len(digits) == 4:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}"
    elif len(digits) == 5:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}"
    elif len(digits) == 6:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}"
    elif len(digits) == 7:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}{digits[6]}"
    elif len(digits) == 8:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}{digits[6]}{digits[7]}"
    elif len(digits) == 9:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}{digits[6]}{digits[7]}{digits[8]}"
    elif len(digits) == 10:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}{digits[6]}{digits[7]}{digits[8]}{digits[9]}"

def format_tunisian_plate_cam_right(texts):
    characters = []
    for text in texts:
        for char in text:
            if char.isalnum():
                characters.append(char)

    digits = [c for c in characters if c.isdigit()]
    # print(f" tous les chiffres: {digits}")

    if len(digits) < 3 or len(digits) > 10:
        return "UNKNOWN"

    if len(digits) == 3:
        return f" TN {digits[0]}{digits[1]}{digits[2]}"
    elif len(digits) == 4:
        return f"TN {digits[0]}{digits[1]}{digits[2]}{digits[3]}"
    elif len(digits) == 5:
        return f"{digits[0]} TN {digits[1]}{digits[2]}{digits[3]}{digits[4]}"
    elif len(digits) == 6:
        return f"{digits[0]}{digits[1]} TN {digits[2]}{digits[3]}{digits[4]}{digits[5]}"
    elif len(digits) == 7:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}{digits[6]}"
    elif len(digits) == 8:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}{digits[6]}{digits[7]}"
    elif len(digits) == 9:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}{digits[6]}{digits[7]}{digits[8]}"
    elif len(digits) == 10:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}{digits[6]}{digits[7]}{digits[8]}{digits[9]}"

def format_tunisian_plate_cam_left(texts):
    characters = []
    for text in texts:
        for char in text:
            if char.isalnum():
                characters.append(char)

    digits = [c for c in characters if c.isdigit()]
    # print(f" tous les chiffres: {digits}")

    if len(digits) < 3 or len(digits) > 7:
        return "UNKNOWN"

    if len(digits) == 3:
        return f"{digits[0]}{digits[1]} TN {digits[2]}"
    elif len(digits) == 4:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}"
    elif len(digits) == 5:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}"
    elif len(digits) == 6:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}"
    elif len(digits) == 7:
        return f"{digits[0]}{digits[1]}{digits[2]} TN {digits[3]}{digits[4]}{digits[5]}{digits[6]}"

# ==============================================================
# Algerian plate formatter
# Expected generic format (simplified):
#   [1-3 digits] [1-4 digits] [2 digits (wilaya)]
# Returns string like: "12 3456 16" or "UNKNOWN" if not plausible
def format_algerian_plate(texts):
    """
    Algerian plate format (post-2009 common format):
      [serial: 5-6 digits] [vehicle info: 3 digits] [wilaya: 2 digits]
        - vehicle info: first digit is vehicle class (1-9), next two are year (00-99)
        - wilaya: 01-58
    Accepts extra leading digits by trimming serial to the last 6.
    Returns formatted string or "UNKNOWN" if not plausible.
    """
    characters = []
    for text in texts:
        for char in text:
            if char.isalnum():
                characters.append(char)

    digits = [c for c in characters if c.isdigit()]
    # Need at least 10 digits to satisfy 6 + 3 + 2 or 5 + 3 + 2
    if len(digits) < 10:
        return "UNKNOWN"

    # Take last 2 as wilaya
    wilaya = "".join(digits[-2:])
    try:
        wilaya_num = int(wilaya)
    except ValueError:
        return "UNKNOWN"
    if not (1 <= wilaya_num <= 58):
        return "UNKNOWN"

    # Previous 3 as vehicle info
    vehicle_info_digits = digits[-5:-2]
    if len(vehicle_info_digits) != 3:
        return "UNKNOWN"
    vehicle_info = "".join(vehicle_info_digits)
    # First must be 1-9
    if vehicle_info[0] not in "123456789":
        return "UNKNOWN"

    # Remainder are serial; keep last 6 if more than 6
    serial_digits = digits[:-5]
    if len(serial_digits) < 5:
        return "UNKNOWN"
    if len(serial_digits) > 6:
        serial_digits = serial_digits[-6:]
    serial = "".join(serial_digits)
    if len(serial) not in (5, 6):
        return "UNKNOWN"

    return f"{serial} {vehicle_info} {wilaya}"

def is_libyan_plate(bg_color: str) -> bool:
    """
    Returns True if the background color is white (Libyan plate), otherwise False.
    """
    return bg_color.strip().lower() == "white"

def format_libyan_plate(region_code: str, serial_number: str) -> str:
    """
    Formats a Libyan license plate as: RegionCode – SerialNumber LYB
    """
    return f"{region_code}-{serial_number} LYB"

def is_libyan_plate_text(plate_text: str) -> bool:


def classify_plate(bg_color: str, ocr_text: str) -> str:
    """
    Classifies the plate based on background color and OCR text.
    - If background is white, format as Libyan plate: first digit is region code, all others are serial number.
    - Otherwise, use existing logic for Algerian/Tunisian.
    """
    if is_libyan_plate(bg_color):
        # Deprecated: use format_libyan_plate_from_digits in main pipeline
        return "LYB"
    else:
        return ocr_text



def get_bg_color(plate_img) -> str:
    """
    Detects the dominant background color of the plate image.
    Returns 'white' if the plate is mostly white, otherwise 'other'.
    """
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    # Expanded white color range in HSV
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = np.sum(mask > 0) / mask.size
    # Lower threshold for white detection
    if white_ratio > 0.4:
        return "white"
    return "other"

def preprocess_plate_img(plate_img):
    """
    Preprocess plate image for better OCR: grayscale, contrast, and adaptive thresholding.
    """
    import cv2
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def classify_plate_with_boxes(bg_color: str, ocr_results) -> str:
    """
    Classifies the plate using bounding box data from EasyOCR.
    - If background is white, format as Libyan plate: first digit (leftmost) is region code, rest are serial number.
    - ocr_results: output from EasyOCR's readtext (list of [bbox, text, conf])
    """
    if is_libyan_plate(bg_color):
        # Extract all digits from all text elements
        all_digits = []
        for bbox, text, conf in ocr_results:
            digits_in_text = re.findall(r'\d', text)
            all_digits.extend(digits_in_text)
        
        if len(all_digits) >= 2:
            region_code = all_digits[0]
            serial_number = ''.join(all_digits[1:])
            formatted_plate = f"{region_code}-{serial_number} LYB"
            return formatted_plate
        elif len(all_digits) == 1:
            return f"{all_digits[0]} LYB"
        else:
            return "LYB"
    else:
        return ' '.join([text for bbox, text, conf in ocr_results])

def format_libyan_plate_from_digits(digits):
    """
    Formats Libyan plate using detected digits list.
    First digit is region code, rest are serial number.
    """
    if len(digits) >= 2:
        region_code = digits[0]
        serial_number = ''.join(digits[1:])
        result = f"{region_code}-{serial_number} LYB"
        return result
    elif len(digits) == 1:
        return f"{digits[0]} LYB"
    else:
        return "LYB"