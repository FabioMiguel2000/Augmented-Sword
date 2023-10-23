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

# Marker size in centimeters
cube_size = 13
# Scale factor
scale_factor = 1

# Define colors for each side
colors = [(0, 0, 255),  # Red
          (0, 255, 0),  # Green
          (255, 0, 0),  # Blue
          (255, 255, 0),  # Yellow
          (0, 255, 255)]  # Cyan

# Array of set of coordinates
blade = []
handle = []

# Sword Blades --------------------------------------
blade.append(
    np.array([
    [-cube_size, -cube_size / 2, cube_size/2],
    [cube_size, -cube_size / 2, cube_size/2],
    [0, cube_size*4, -cube_size/2]
], dtype=np.float32))
blade.append(
    np.array([
    [-cube_size, -cube_size / 2, -(cube_size + cube_size/2)],
    [cube_size, -cube_size / 2, -(cube_size + cube_size/2)],
    [0, cube_size*4, -cube_size/2]
], dtype=np.float32))
blade.append(
    np.array([
    [-cube_size, -cube_size / 2, -(cube_size + cube_size/2)],
    [-cube_size, -cube_size / 2, cube_size/2],
    [0, cube_size*4, -cube_size/2]
], dtype=np.float32))
blade.append(
    np.array([
    [cube_size, -cube_size / 2, cube_size/2],
    [cube_size, -cube_size / 2, -(cube_size + cube_size/2)],
    [0, cube_size*4, -cube_size/2]
], dtype=np.float32))

# Sword Handle --------------------------------------
handle.append(
    np.array([
    # Top Face
    [-cube_size/4, -cube_size*2, -cube_size/4],
    [cube_size/4, -cube_size*2, -cube_size/4],
    [cube_size/4, -cube_size*2, -3*cube_size/4],
    [-cube_size/4, -cube_size*2, -3*cube_size/4],
], dtype=np.float32))
handle.append(
    np.array([
    # Bottom Face
    [-cube_size/4, -5*cube_size/2, -cube_size/4],
    [cube_size/4, -5*cube_size/2, -cube_size/4],
    [cube_size/4, -5*cube_size/2, -3*cube_size/4],
    [-cube_size/4, -5*cube_size/2, -3*cube_size/4],
], dtype=np.float32))
handle.append(
    np.array([
    # Left Face
    [-cube_size/4, -cube_size*2, -3*cube_size/4],
    [-cube_size/4, -5*cube_size/2, -3*cube_size/4],
    [-cube_size/4, -5*cube_size/2, -cube_size/4],
    [-cube_size/4, -cube_size*2, -cube_size/4],
], dtype=np.float32))
handle.append(
    np.array([
    # Right Face
    [cube_size/4, -cube_size*2, -cube_size/4],
    [cube_size/4, -5*cube_size/2, -cube_size/4],
    [cube_size/4, -5*cube_size/2, -3*cube_size/4],
    [cube_size/4, -cube_size*2, -3*cube_size/4],
], dtype=np.float32))
handle.append(
    np.array([
    # Front Face
    [-cube_size/4, -cube_size*2, -cube_size/4],
    [-cube_size/4, -5*cube_size/2, -cube_size/4],
    [cube_size/4, -5*cube_size/2, -cube_size/4],
    [cube_size/4, -cube_size*2, -cube_size/4],
], dtype=np.float32))
handle.append(
    np.array([
    # Back Face
    [-cube_size/4, -cube_size*2, -3*cube_size/4],
    [-cube_size/4, -5*cube_size/2, -3*cube_size/4],
    [cube_size/4, -5*cube_size/2, -3*cube_size/4],
    [cube_size/4, -cube_size*2, -3*cube_size/4],
], dtype=np.float32))
top_marker_translation = [0, -cube_size/2, -cube_size]

# Scale the shapes
for i in range(len(handle)):
    handle[i] *= scale_factor
for i in range(len(blade)):
    blade[i] *= scale_factor

# Compute necessary transformations for the top marker
def get_top_shape(shape, rotation_matrix):
    coordinates = shape
    # Apply the rotation to the coordinates
    rotated_coordinates = np.dot(coordinates, rotation_matrix.T)
    new_shape = rotated_coordinates + top_marker_translation
    return new_shape

def draw_poly(shape, rvec, tvec, cameraMatrix, distCoeffs, color):
    projected_shape, _ = cv2.projectPoints(shape, rvec, tvec, cameraMatrix, distCoeffs)
    projected_shape = np.int32(projected_shape).reshape(-1, 2)
    cv2.fillPoly(frame, [projected_shape], color=color)

def draw_polylines(shape, rvec, tvec, cameraMatrix, distCoeffs, color):
    projected_shape, _ = cv2.projectPoints(shape, rvec, tvec, cameraMatrix, distCoeffs)
    projected_shape = np.int32(projected_shape).reshape(-1, 2)
    cv2.polylines(frame, [projected_shape], color=color, isClosed=True, thickness=2)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        print("\n-------------------")
        print("IDs detected:    ", ids.flatten())

        # Compute distances to each marker
        distances = []
        vecs = []
        for i in range(len(ids)):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i].reshape(corners[i].shape), 10, cameraMatrix, distCoeffs)
            distance = np.sqrt(tvec[0][0][2] ** 2 + tvec[0][0][0] ** 2 + tvec[0][0][1] ** 2)
            distances.append(distance)
            vecs.append((rvec, tvec))

        print("Distances:       ", distances)
            
        # Find the closest marker
        id_index = np.argmin(distances)
        id = ids.flatten()[id_index]
        print("ID used:         ", id)

        # Get vecs of closest marker
        rvec, tvec = vecs[id_index]
        if (id == 0):
            rvec2 = np.array([np.pi/2, 0, 0])
            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec2)

            # Draw sword (handle and blade)
            for shape in handle:
                shape = get_top_shape(shape, rotation_matrix)
                draw_poly(shape, rvec, tvec, cameraMatrix, distCoeffs, colors[0])
            for shape in handle:
                shape = get_top_shape(shape, rotation_matrix)
                draw_polylines(shape, rvec, tvec, cameraMatrix, distCoeffs, colors[4])
            for shape in blade:
                shape = get_top_shape(shape, rotation_matrix)
                draw_poly(shape, rvec, tvec, cameraMatrix, distCoeffs, colors[1])
            for shape in blade:
                shape = get_top_shape(shape, rotation_matrix)
                draw_polylines(shape, rvec, tvec, cameraMatrix, distCoeffs, colors[4])

        else:
            # Draw sword (handle and blade)
            for shape in handle:
                draw_poly(shape, rvec, tvec, cameraMatrix, distCoeffs, colors[0])
            for shape in handle:
                draw_polylines(shape, rvec, tvec, cameraMatrix, distCoeffs, colors[4])
            for shape in blade:
                draw_poly(shape, rvec, tvec, cameraMatrix, distCoeffs, colors[1])
            for shape in blade:
                draw_polylines(shape, rvec, tvec, cameraMatrix, distCoeffs, colors[4])
            
    aruco.drawDetectedMarkers(frame, corners)
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
