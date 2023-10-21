import cv2
import cv2.aruco as aruco
import numpy as np

# Enable marker IDs for recognition
ENABLED_IDS = [0, 1, 2, 3, 4]

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

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        for i in range(len(ids)):
            if ids[i] in ENABLED_IDS:
                # Define the 3D sword representation in object coordinates
                # Make sure it's in the correct data type (e.g., float32)
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

                # Estimate pose for the current marker
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 10, cameraMatrix, distCoeffs)

                # Project the 3D points to the 2D image plane
                imagePoints, _ = cv2.projectPoints(sword_3d, rvec, tvec, cameraMatrix, distCoeffs)

                # Draw the wireframe of the 3D sword
                for j in range(len(imagePoints)):
                    pt1 = tuple(map(int, imagePoints[j - 1].ravel()))
                    pt2 = tuple(map(int, imagePoints[j].ravel()))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            aruco.drawDetectedMarkers(frame, corners)

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


