import cv2
import numpy as np
from marker_detection import test


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
cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
distCoeffs = np.array([k1, k2, p1, p2, k3])

# Define a threshold for black pixels
black_threshold = [30, 30, 30]

# Read sword image
img = cv2.imread('../img/sword.png')
assert img is not None, "Image could not be read, check with os.path.exists()"
# Downscale the image
down_width = img.shape[1]//4
down_height = img.shape[0]//4
down_points = (down_width, down_height)
img = cv2.resize(img, (down_width,down_height), interpolation= cv2.INTER_LINEAR)

# Define the marker's original corners (ID: 0)
marker_orig_width = 100
y_offset = (img.shape[0] - marker_orig_width)//2
x_offset = (img.shape[1] - marker_orig_width)//2
marker_orig = np.float32([[x_offset, y_offset], [x_offset+marker_orig_width, y_offset],[x_offset+marker_orig_width, y_offset+marker_orig_width],[x_offset, y_offset+marker_orig_width]])
# Rotate marker_orig by ~120 degrees
rotation_matrix = cv2.getRotationMatrix2D((x_offset + marker_orig_width/2, y_offset + marker_orig_width/2), 114 + 180, 1)
marker_orig = cv2.transform(marker_orig.reshape(-1, 1, 2), rotation_matrix).reshape(4, 2)

# Define the marker's original corners (ID: 1)
y_offset = y_offset
x_offset = x_offset
marker_orig1 = np.float32([[x_offset, y_offset], [x_offset+marker_orig_width, y_offset],[x_offset+marker_orig_width, y_offset+marker_orig_width],[x_offset, y_offset+marker_orig_width]])
# Rotate marker_orig1 by ~(-120) degrees
rotation_matrix = cv2.getRotationMatrix2D((x_offset + marker_orig_width/2, y_offset + marker_orig_width/2), -114 + 180, 1)
marker_orig1 = cv2.transform(marker_orig1.reshape(-1, 1, 2), rotation_matrix).reshape(4, 2)

while True:

    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # corners, _ = detect_marker(frame=frame, marker_path='../img/samples/marker_1.png')
    frame = test(frame, '../img/samples/marker_1.png')
    # # print(contours)
    # if corners != [] :
    #     corners = np.array(corners, dtype=np.int32)

    #     # Find the extreme points (minimum and maximum x and y)
    #     min_x = np.min(corners[:, 0])
    #     max_x = np.max(corners[:, 0])
    #     min_y = np.min(corners[:, 1])
    #     max_y = np.max(corners[:, 1])

    #     # Create a new array with ordered corner points
    #     ordered_corner_points = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], dtype=np.int32)

    #     # Reshape the corner points for OpenCV
    #     # print(contours)
        
    #     # cv2.polylines(frame, [ordered_corner_points], isClosed=True, color=(0, 255, 0), thickness=2)
    #     # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    #     # Create a mask image and draw a filled rectangle for the bounding box
    #     # Create an empty mask with the same size as the image
    #     mask = np.zeros_like(frame)

    #     # Fill the polygon defined by the corner points with white (255)
    #     cv2.fillPoly(mask, [ordered_corner_points], (255, 255, 255))

    #     # Convert the mask to grayscale
    #     gray_mask = cv2.cvtCpythoolor(mask, cv2.COLOR_BGR2GRAY)

    #     # Find the contour of the filled polygon
    #     contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # Draw the contour on the original image
    #     cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()