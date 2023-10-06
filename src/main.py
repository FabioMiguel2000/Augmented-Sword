import cv2
import cv2.aruco as aruco
import numpy as np
from constants import *

ENABLED_IDS = [0, 1]

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

# Get the predefined ArUco dictionary (you can use different dictionaries)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Provide your estimated calibration values here (replace placeholders)
# fx = 600.0
# fy = 600.0
# cx = frame_width / 2
# cy = frame_height / 2
# k1 = 0.0
# k2 = 0.0
# p1 = 0.0
# p2 = 0.0
# k3 = 0.0

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
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert the frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)

    # Draw a 3D cube with edges and filled area on top of detected markers (if any)
    if ids is not None:
        for i in range(len(ids)):
            if (ids[i] not in ENABLED_IDS): continue
            #print("IDs:", ids[i])

            if (ids[i] == 1):
                # Compute the affine transformation
                M = cv2.getAffineTransform(marker_orig1[1:4], corners[0].reshape(4, 2)[1:4])
                overlay_warped_flipped = cv2.warpAffine(cv2.flip(img,1), M, (frame_width, frame_height))
                overlay_warped = overlay_warped_flipped
            else:
                # Compute the affine transformation
                M = cv2.getAffineTransform(marker_orig[1:4], corners[0].reshape(4, 2)[1:4])
                overlay_warped = cv2.warpAffine(img, M, (frame_width, frame_height))

            aruco.drawDetectedMarkers(frame, corners)

            # Find the mask of non-black pixels in overlay_warped
            non_black_mask = np.all(overlay_warped > black_threshold, axis=-1)
            # Use the mask to replace pixels in frame
            frame[non_black_mask] = overlay_warped[non_black_mask]

    # Display the frame in the "Webcam Feed" window
    cv2.imshow("Webcam Feed", frame)

    # Check for the 'q' key to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()