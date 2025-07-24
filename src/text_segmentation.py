import cv2

def segment_characters(image_path):
    img = cv2.imread(image_path, 0)
    
    # Use the preprocessed binary image for segmentation
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        roi = img[y:y+h, x:x+w]
        cv2.imwrite(f'output/char_{idx}.jpg', roi)

if __name__ == "__main__":
    segment_characters('output/preprocessed_image.jpg')
