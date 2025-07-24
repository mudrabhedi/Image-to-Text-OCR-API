import cv2
import pytesseract

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def pre_processing(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error loading image.")
        return None
    else:
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply moderate denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=15)

        # Step 3: Increase contrast for better text clarity
        contrast_enhanced = cv2.convertScaleAbs(denoised, alpha=2.0, beta=0)

        # Save the final preprocessed image
        output_path = 'output/preprocessed_image.jpg'
        cv2.imwrite(output_path, contrast_enhanced)

        return output_path

def text_extraction(output_path):
    img_for_ocr = cv2.imread(output_path)
    if img_for_ocr is None:
        print("Error loading image for text recognition.")
        return None
    else:
        # Tesseract OCR with custom config
        config = "--oem 3 --psm 6"
        recognized_text = pytesseract.image_to_string(img_for_ocr, config=config)
        
        return recognized_text.strip()

def main(image_path):
    print("Starting text detection...")
    
    # Step 1: Preprocess the image
    preprocessed_image_path = pre_processing(image_path)
    if not preprocessed_image_path:
        print("Preprocessing failed.")
        return

    print("Preprocessing completed.")
    
    # Step 2: Detect text
    detected_text = text_extraction(preprocessed_image_path)
    if detected_text:
        print("Detected Text:")
        print(detected_text)
    else:
        print("No text detected.")
    print('Done.')

if __name__ == "__main__":
    main(r'c:\DIP PROJECT\dip mini project\images\image3.png')  # Adjust the image path accordingly
