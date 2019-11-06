# Face Detection with OpenCV

# Import OpenCV
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Face Detector
def face_detector(grayScale, originalFrame):
    faces = face_cascade.detectMultiScale(grayScale, 1.3, 5) # Coordinates of detected faces
    for (x, y, w, h) in faces: # For loop to iterate over detected faces
        cv2.rectangle(originalFrame, 
                      (x, y), # Upper Left Corner
                      (x+w, y+h), # Lower Right Corner 
                      (0, 255, 0), 3 # Thickness) # Drawing rectangle over detected face
        
        roi_gray = grayScale[y:y+h, x:x+w] # Region of Interest for Grayscale Image
        roi_color = originalFrame[y:y+h, x:x+w] # Region of Interest for Original Image
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # Coordinates of detected eyes