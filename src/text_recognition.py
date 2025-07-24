import cv2
import pytesseract

def recognize_text(image_path):
    # Ensure Tesseract is installed and the path is configured
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this as needed
    
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error loading image for text recognition.")
        return ""
    
    # Use Tesseract to recognize text
    text = pytesseract.image_to_string(img, config='--psm 6')  # psm 6 works well for uniform blocks of text
    
    return text
