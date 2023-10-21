import cv2
import cv2.aruco as aruco
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Feed", frame_width, frame_height)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Camera parameters (You may need to adjust these)
fx = 600.0
fy = 600.0
cx = frame_width / 2
cy = frame_height / 2
cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
distCoeffs = np.zeros((5, 1))

# Cube size in centimeters
cube_size = 7.0

# Define the 3D coordinates of the markers on the cube based on the cube size
marker_coordinates_3d = [
    np.array([[-cube_size / 2, -cube_size / 2, 0.0]]),    # Top marker (ID: 0)
    np.array([[0.0, -cube_size / 2, cube_size / 2]]),     # Front marker (ID: 1)
    np.array([[-cube_size / 2, 0.0, cube_size / 2]]),     # Right marker (ID: 2)
    np.array([[-cube_size / 2, -cube_size / 2, cube_size]]),  # Back marker (ID: 3)
    np.array([[-cube_size, -cube_size / 2, cube_size / 2]])  # Left marker (ID: 4)
]


# Define colors for each side
colors = [(0, 0, 255),  # Red
          (0, 255, 0),  # Green
          (255, 0, 0),  # Blue
          (255, 255, 0),  # Yellow
          (0, 255, 255)]  # Cyan

# Define the 3D coordinates of the sword
sword_3d = np.array([
    [0.0, 0.0, 0.0],
    [20.0, 0.0, 0.0],
    [20.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [20.0, 0.0, -1.0],
    [20.0, 1.0, -1.0],
    [0.0, 1.0, -1.0]
], dtype=np.float32)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        for i in range(len(ids)):
            if ids[i] in range(5):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 10, cameraMatrix, distCoeffs)
                marker_id = int(ids[i][0])  # Convert to integer

                # Draw the filled shape of the sword with the assigned color
                sword_points, _ = cv2.projectPoints(sword_3d, rvec, tvec, cameraMatrix, distCoeffs)
                sword_points = np.int32(sword_points).reshape(-1, 2)
                cv2.fillPoly(frame, [sword_points], color=colors[marker_id])

    aruco.drawDetectedMarkers(frame, corners)

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
