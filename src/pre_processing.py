import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error loading image.")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to enhance text
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Save the preprocessed image
    output_path = 'output/preprocessed_image.jpg'
    cv2.imwrite(output_path, binary)
    
    return output_path
