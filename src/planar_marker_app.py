import cv2
import numpy as np
from marker_detection import detect_marker_on_frame

import cv2
import numpy as np

# Initialize the webcam (you may need to change the camera index if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the width and height of the frame (you can adjust these values as needed)
frame_width = 640
frame_height = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Create a window to display the webcam feed
cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Feed", frame_width, frame_height)

try: 
    from constants import *
except:
    print("constants.py not found, using default values")
    fx = 600.0
    fy = 600.0
    cx = frame_width / 2
    cy = frame_height / 2
    k1 = 0.0
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0
    k3 = 0.0

# Construct the camera matrix and distortion coefficients
cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

marker_1 = cv2.imread('../img/samples/marker_1.png')
marker_0 = cv2.imread('../img/samples/marker_0.png')
while True:

    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    frame = detect_marker_on_frame(frame, [marker_0, marker_1], cameraMatrix, distCoeffs)

    # Display the result image
    cv2.imshow("Webcam Feed", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()