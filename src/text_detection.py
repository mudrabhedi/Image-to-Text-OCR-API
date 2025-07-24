import cv2

def detect_text_regions(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error loading image for text detection.")
        return None
    
    # Assuming the entire image is text (adjust as needed for more complex use cases)
    
    # Optionally, you can apply additional image processing techniques like edge detection
    
    output_path = 'output/detected_text_regions.jpg'
    cv2.imwrite(output_path, img)  # Save detected text regions (same as preprocessed image)
    
    return output_path
