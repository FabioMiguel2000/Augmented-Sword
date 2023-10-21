import cv2
import numpy as np
from marker_detection import detect_marker_on_frame

import cv2
import numpy as np

# Load the image in which you want to detect the marker
image = cv2.imread("image_with_marker.jpg")

def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# Function to draw a 3D pyramid on the marker
def draw_pyramid(image, marker_corners):
    pyramid_height = 200
    if len(marker_corners) == 4:
        apex = (int((marker_corners[0][0] + marker_corners[2][0]) / 2),
                int((marker_corners[0][1] + marker_corners[2][1]) / 2) - pyramid_height)

        # Draw the edges of the 3D pyramid using cv2.line()
        for i in range(4):
            point1 = marker_corners[i]
            point2 = marker_corners[(i + 1) % 4]
            cv2.line(image, point1, point2, (0, 255, 0), 2)

            # Connect each marker corner to the apex of the pyramid
            cv2.line(image, marker_corners[i], apex, (0, 255, 0), 2)

        # Connect the apex to form the base of the pyramid
        for i in range(4):
            cv2.line(image, apex, marker_corners[i], (0, 255, 0), 2)

    return image

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

    original_marker = cv2.imread('../img/samples/marker_1.png')
    frame, marker_corners = detect_marker_on_frame(frame, original_marker)

    # print(marker_corners)
    if len(marker_corners) != 0:
        marker_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)

        centroid = np.mean(marker_corners, axis=0)

        # Calculate the distance of each corner from the centroid
        distances = [np.linalg.norm(corner - centroid) for corner in marker_corners]

        # Find the indices of the corners with the maximum and minimum distances
        top_left_index = np.argmin(distances)
        bottom_right_index = np.argmax(distances)

        # Remove the top left and bottom right corners from the original list
        remaining_corners = [corner for i, corner in enumerate(marker_corners) if i != top_left_index and i != bottom_right_index]

        # Find the top right and bottom left corners from the remaining two corners
        top_right_corner, bottom_left_corner = remaining_corners

        # Now you have separated the corners into top left, top right, bottom left, and bottom right
        print("Top Left:", marker_corners[top_left_index])
        print("Top Right:", top_right_corner)
        print("Bottom Left:", bottom_left_corner)
        print("Bottom Right:", marker_corners[bottom_right_index])

        print("marker coordinates: " , marker_corners)

        # homography_matrix, _ = cv2.findHomography(marker_corners, marker_3d)

         # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv2.solvePnP(marker_3d, marker_corners, cameraMatrix, distCoeffs)

        print("RVEC = ", rvecs)
        print("TVECS = ", tvecs)

        # Render the X, Y, and Z axes on the marker
        # You can draw lines or 3D objects using the rotation and translation information

        # Display the result
        cv2.imshow('Marker Pose Estimation', frame)



        # Display the result image
    cv2.imshow("Webcam Feed", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()