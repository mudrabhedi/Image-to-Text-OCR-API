import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def upscale_image(img, scale_factor=2):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Error loading image.")

        # Upscale for better recognition
        img = upscale_image(img)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # Dilate to join letters
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)

        # Apply median blur to reduce noise
        blurred = cv2.medianBlur(dilated, 3)

        # Save the preprocessed image for debugging
        output_path = 'output/preprocessed_image.jpg'
        cv2.imwrite(output_path, blurred)

        return output_path

    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None


def detect_text_regions(image_path):
    try:
        # Read the preprocessed image (which should be a single-channel image)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is None:
            raise Exception("Error loading image for text detection.")

        # Find contours on the preprocessed grayscale image
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank image to draw rectangles
        img_contours = np.zeros_like(img)  # Create a blank image of the same size

        # Loop over contours and filter out small ones
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Adjust size filter to capture smaller text
            if w > 10 and h > 10:  # Reduced size threshold
                cv2.rectangle(img_contours, (x, y), (x + w, y + h), 255, 2)  # Draw white rectangles

        # Save the image with detected text regions highlighted
        output_path = 'output/detected_text_regions.jpg'
        cv2.imwrite(output_path, img_contours)

        return output_path

    except Exception as e:
        print(f"Error detecting text regions: {str(e)}")
        return None

def recognize_text(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Error loading image for text recognition.")

        # Tesseract configurations to improve text recognition
        custom_config = r'--oem 1 --psm 6'  # Experiment with different PSM values

        # Recognize text
        text = pytesseract.image_to_string(img, config=custom_config)

        return text

    except Exception as e:
        print(f"Error recognizing text: {str(e)}")
        return None

def main(image_path):
    print("Starting text detection...")

    # Step 1: Preprocess the image
    preprocessed_image_path = preprocess_image(image_path)
    if not preprocessed_image_path:
        print("Preprocessing failed.")
        return

    print("Preprocessing completed . Proceeding to text detection.")

    # Step 2: Detect text regions
    text_region_image_path = detect_text_regions(preprocessed_image_path)
    if not text_region_image_path:
        print("Text region detection failed.")
        return

    # Step 3: Recognize text from detected regions
    recognized_text = recognize_text(text_region_image_path)
    if recognized_text.strip():
        print("Text Extraction Completed Successfully!")
        print(f"Extracted text:\n{recognized_text}")
    else:
        print("No text detected.")

if __name__ == "__main__":
    main(r'C:\DIP PROJECT\dip mini project\images\image1.jpeg')  # Adjust the image path accordingly
    