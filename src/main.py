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

# Define the 3D coordinates of the center of the markers on the cube based on the cube size
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


# Old Sword
sword_3d_1 = np.array([
    [0.0, 0.0, 0.0],
    [20.0, 0.0, 0.0],
    [20.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [20.0, 0.0, -1.0],
    [20.0, 1.0, -1.0],
    [0.0, 1.0, -1.0],
    [8.0, 0.0, 0.0],   # Handle - Top
    [12.0, 0.0, 0.0],  # Handle - Top
    [10.0, 0.0, 0.0],  # Handle - Middle (Center)
    [10.0, 0.0, -4.0],  # Handle - Bottom
], dtype=np.float32)

# # Sword Blades --------------------------------------
# blade_front = np.array([
#     [-8, -8, 1],
#     [8, -8, 1],
#     [0, 90, -3.5],
# ], dtype=np.float32)
# blade_back = np.array([
#     [-8,-8,-18],
#     [8,-8,-18],
#     [0, 90, -3.5],
# ], dtype=np.float32)
# blade_left = np.array([
#     [-8, -8, -18],
#     [-8, -8, 1],
#     [0, 90, -3.5],
# ], dtype=np.float32)
# blade_right = np.array([
#     [8, -8, -18],
#     [8, -8, 1],
#     [0, 90, -3.5],
# ], dtype=np.float32)

# Sword Blades --------------------------------------
blade_front = np.array([
    [-cube_size, -cube_size / 2, cube_size/2],
    [cube_size, -cube_size / 2, cube_size/2],
    [0, cube_size*3, -cube_size/2]
], dtype=np.float32)
blade_back = np.array([
    [-cube_size, -cube_size / 2, -(cube_size + cube_size/2)],
    [cube_size, -cube_size / 2, -(cube_size + cube_size/2)],
    [0, cube_size*3, -cube_size/2]
], dtype=np.float32)
blade_left = np.array([
    [-cube_size, -cube_size / 2, -(cube_size + cube_size/2)],
    [-cube_size, -cube_size / 2, cube_size/2],
    [0, cube_size*3, -cube_size/2]
], dtype=np.float32)
blade_right = np.array([
    [cube_size, -cube_size / 2, cube_size/2],
    [cube_size, -cube_size / 2, -(cube_size + cube_size/2)],
    [0, cube_size*3, -cube_size/2]
], dtype=np.float32)

# Sword Handle --------------------------------------
lower_handle_top = np.array([
    # Top Face
    [-2, -30, -4],
    [2, -30, -4],
    [2, -30, -8],
    [-2, -30, -8],
], dtype=np.float32)
lower_handle_bottom = np.array([
    # Bottom Face
    [-2, -35, -4],
    [2, -35, -4],
    [2, -35, -8],
    [-2, -35, -8],
], dtype=np.float32)
lower_handle_left = np.array([
    # Left Face
    [-2, -30, -4],
    [-2, -35, -4],
    [-2, -35, -8],
    [-2, -30, -8],
], dtype=np.float32)
lower_handle_right = np.array([
    # Right Face
    [2, -30, -4],
    [2, -35, -4],
    [2, -35, -8],
    [2, -30, -8],
], dtype=np.float32)
lower_handle_front = np.array([
    # Front Face
    [-2, -30, -4],
    [2, -30, -4],
    [2, -35, -4],
    [-2, -35, -4],
], dtype=np.float32)
lower_handle_back = np.array([
    # Back Face
    [-2, -30, -8],
    [2, -30, -8],
    [2, -35, -8],
    [-2, -35, -8],
], dtype=np.float32)

#lower_handle = np.vstack((lower_handle_top, lower_handle_bottom, lower_handle_left, lower_handle_right, lower_handle_front, lower_handle_back))



scale_factor = 1.3
#sword = np.vstack((blade_1, blade_2)) * scale_factor

sword_front = blade_front * scale_factor
sword_back = blade_back * scale_factor
sword_left = blade_left * scale_factor
sword_right = blade_right * scale_factor

lower_handle_top = lower_handle_top * scale_factor
lower_handle_bottom = lower_handle_bottom * scale_factor
lower_handle_left = lower_handle_left * scale_factor
lower_handle_right = lower_handle_right * scale_factor
lower_handle_front = lower_handle_front * scale_factor
lower_handle_back = lower_handle_back * scale_factor


#sword_lower_handle = lower_handle * scale_factor

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        if (ids[0] != 0):
            for id in ids[0]:
                if (id != 2):
                    continue
                print("\n-------------------")
                print("IDs detected:    ", len(ids), "/ 3")
                print("ID used:         ", ids[0][0])
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], 10, cameraMatrix, distCoeffs)
                marker_id = int(ids[0][0])  # Convert to integer

                # Draw the filled shape of the sword with the assigned color
                sword_points_back, _ = cv2.projectPoints(sword_back, rvec, tvec, cameraMatrix, distCoeffs)
                sword_points_back = np.int32(sword_points_back).reshape(-1, 2)
                cv2.fillPoly(frame, [sword_points_back], color=colors[1])

                sword_points_front, _ = cv2.projectPoints(sword_front, rvec, tvec, cameraMatrix, distCoeffs)
                sword_points_front = np.int32(sword_points_front).reshape(-1, 2)
                cv2.fillPoly(frame, [sword_points_front], color=colors[1])

                sword_points_left, _ = cv2.projectPoints(sword_left, rvec, tvec, cameraMatrix, distCoeffs)
                sword_points_left = np.int32(sword_points_left).reshape(-1, 2)
                cv2.fillPoly(frame, [sword_points_left], color=colors[1])

                sword_points_right, _ = cv2.projectPoints(sword_right, rvec, tvec, cameraMatrix, distCoeffs)
                sword_points_right = np.int32(sword_points_right).reshape(-1, 2)
                cv2.fillPoly(frame, [sword_points_right], color=colors[1])

                # # Draw the filled shape of the lower handle with the assigned color
                # lower_handle_points_top, _ = cv2.projectPoints(lower_handle_top, rvec, tvec, cameraMatrix, distCoeffs)
                # lower_handle_points_top = np.int32(lower_handle_points_top).reshape(-1, 2)
                # cv2.fillPoly(frame, [lower_handle_points_top], color=colors[2])

                # lower_handle_points_bottom, _ = cv2.projectPoints(lower_handle_bottom, rvec, tvec, cameraMatrix, distCoeffs)
                # lower_handle_points_bottom = np.int32(lower_handle_points_bottom).reshape(-1, 2)
                # cv2.fillPoly(frame, [lower_handle_points_bottom], color=colors[2])

                # lower_handle_points_left, _ = cv2.projectPoints(lower_handle_left, rvec, tvec, cameraMatrix, distCoeffs)
                # lower_handle_points_left = np.int32(lower_handle_points_left).reshape(-1, 2)
                # cv2.fillPoly(frame, [lower_handle_points_left], color=colors[2])

                # lower_handle_points_right, _ = cv2.projectPoints(lower_handle_right, rvec, tvec, cameraMatrix, distCoeffs)
                # lower_handle_points_right = np.int32(lower_handle_points_right).reshape(-1, 2)
                # cv2.fillPoly(frame, [lower_handle_points_right], color=colors[2])

                # lower_handle_points_front, _ = cv2.projectPoints(lower_handle_front, rvec, tvec, cameraMatrix, distCoeffs)
                # lower_handle_points_front = np.int32(lower_handle_points_front).reshape(-1, 2)
                # cv2.fillPoly(frame, [lower_handle_points_front], color=colors[2])

                # lower_handle_points_back, _ = cv2.projectPoints(lower_handle_back, rvec, tvec, cameraMatrix, distCoeffs)
                # lower_handle_points_back = np.int32(lower_handle_points_back).reshape(-1, 2)
                # cv2.fillPoly(frame, [lower_handle_points_back], color=colors[2])

    aruco.drawDetectedMarkers(frame, corners)

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
