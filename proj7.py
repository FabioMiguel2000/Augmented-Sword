import cv2
import cv2.aruco as aruco
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

# Get the predefined ArUco dictionary (you can use different dictionaries)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Define the 3D cube coordinates with equal edge lengths
# You can adjust the scale factor as needed
edge_length = 0.1  # Adjust this value to set the size of the cube
cube_pts = np.array([[-edge_length / 2, -edge_length / 2, 0],
                     [edge_length / 2, -edge_length / 2, 0],
                     [edge_length / 2, edge_length / 2, 0],
                     [-edge_length / 2, edge_length / 2, 0],
                     [-edge_length / 2, -edge_length / 2, -edge_length],
                     [edge_length / 2, -edge_length / 2, -edge_length],
                     [edge_length / 2, edge_length / 2, -edge_length],
                     [-edge_length / 2, edge_length / 2, -edge_length]])

# Define cube edges
cube_edges = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

# Define cube faces (indices of cube_pts for each face)
cube_faces = [[0, 1, 2, 3], [4, 5, 6, 7],
              [0, 1, 5, 4], [1, 2, 6, 5],
              [2, 3, 7, 6], [3, 0, 4, 7]]

# Provide your estimated calibration values here (replace placeholders)
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

# Define colors for the edges and filled area
edge_color = (0, 0, 255)  # Red color for edges
fill_color = (0, 255, 0)  # Green color for filled area

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Webcam frame not read properly.")
        break

    # Convert the frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)

    # Draw a 3D cube with edges and filled area on top of detected markers (if any)
    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs)
            rvec, tvec = rvec.reshape((3, 1)), tvec.reshape((3, 1))
            imgpts, _ = cv2.projectPoints(cube_pts, rvec, tvec, cameraMatrix, distCoeffs)

            for face in cube_faces:
                face_pts = imgpts[face].astype(int)
                cv2.fillPoly(frame, [face_pts], fill_color)

            for edge in cube_edges:
                pt1 = tuple(map(int, imgpts[edge[0]].ravel()))
                pt2 = tuple(map(int, imgpts[edge[1]].ravel()))
                cv2.line(frame, pt1, pt2, edge_color, 2)

            aruco.drawDetectedMarkers(frame, corners)

    # Display the frame in the "Webcam Feed" window
    cv2.imshow("Webcam Feed", frame)

    # Check for the 'q' key to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
