import cv2
import numpy as np

image_path = '../img/samples/examples/example_2.png'
marker_path = '../img/samples/marker_1.png'

image = cv2.imread(image_path)
marker = cv2.imread(marker_path)

frame_height, frame_width, _ = image.shape

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

# Define a function to calculate the angle between two vectors
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    return np.arccos(dot_product / (magnitude_v1 * magnitude_v2))

def image_binarization(image, THRESHOLD_VALUE = 128, USE_OTSU_METHOD = 1, INVERTED = 0):
    # Apply gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if USE_OTSU_METHOD == 1:
        # Apply Otsu's thresholding
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    else:
        # Binarization (Thresholding)
        _, binary_image = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    if INVERTED == 1:
        inverted_image = cv2.bitwise_not(binary_image)
        return inverted_image

    return binary_image

def connected_components(binary_image, image, ASPECT_RATIO_MIN = 0.7, ASPECT_RATIO_MAX = 1.45, MIN_AREA_THRESHOLD = 2000):
    # `num_labels` gives the total number of labeled regions
    # `labeled_image` is an image with each pixel labeled with its region's ID
    # `stats` is a NumPy array containing statistics for each labeled region
    # `centroids` contains the centroids of each labeled region
    # cv2.imshow('Labeled', labeled_image)
    num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Create a colored label image
    colored_label_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)

    # Uncomment this to show colored label image
    # cv2.imshow("Colored Label Image ", colored_label_image)

    # Create a list to store the images and contours of each component
    component_images = []

    # The following loop will filter out (small regions based on area) AND (non-rectangular regions)
    for label in range(1, num_labels):

        x, y, w, h, area = stats[label]
        aspect_ratio = float(w) / h
        
        # Filter out small regions based on area and non-rectangular regions
        if stats[label, cv2.CC_STAT_AREA] >= MIN_AREA_THRESHOLD and ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX:
            
            # Generate a random color for each label
            color = np.random.randint(0, 255, size=3)
            
            # Set pixels with the current label to the chosen color
            colored_label_image[labeled_image == label] = color

            component_mask = (labeled_image == label).astype(np.uint8)
    
            # Apply the mask to the original image to extract the component
            component_image = cv2.bitwise_and(image, image, mask=component_mask)
            
            # Store the component image
            component_images.append(component_image)

    return component_images


def find_contours(image):
    # Convert the component image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the contours in the edge-detected image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def find_corners_from_countours(image, contours):
    # Initialize a list to store the filtered corners
    group_of_corners = []

    # Iterate through the detected contours and filter corners using quadrilateral fitting
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            # Ensure that the approximated contour is a quadrilateral
            approx = approx.reshape(-1, 2)
            
            # Calculate the angles between the sides of the quadrilateral
            angles = []
            for i in range(4):
                v1 = approx[(i + 1) % 4] - approx[i]
                v2 = approx[(i + 2) % 4] - approx[(i + 1) % 4]
                angle = angle_between_vectors(v1, v2)
                angles.append(angle)
            
            # Define a threshold for angle differences (e.g., 10 degrees)
            max_angle_diff = np.deg2rad(30)
            
            # Check if opposite sides are approximately parallel
            if np.abs(angles[0] - angles[2]) < max_angle_diff and np.abs(angles[1] - angles[3]) < max_angle_diff:
                group_of_corners.append([approx])

    print("Relevant Connected Components found = ", len(group_of_corners))

    return group_of_corners


def estimate_pose_single_marker(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    
    marker_points = np.array([[0, 0, 0],
                            [marker_size, 0, 0],
                            [marker_size, marker_size , 0],
                            [0, marker_size, 0]], dtype=np.float32)

    corners = np.array(corners, dtype=np.float32)

    _, R, t = cv2.solvePnP(marker_points, corners, mtx, distortion)

    rvecs = np.array(R, dtype=np.float32)
    tvecs = np.array(t, dtype=np.float32)

    return rvecs, tvecs

def render_cuboid(frame ,cuboid_object, rvecs, tvecs, cameraMatrix, distCoeffs):
    points, _ = cv2.projectPoints(cuboid_object, rvecs, tvecs, cameraMatrix, distCoeffs)

    for i in range(4):
        cv2.line(frame, (int(points[i][0][0]), int(points[i][0][1])), (int(points[i + 4][0][0]), int(points[i + 4][0][1])), (0, 255, 0), 2)
        next_index = (i + 1) if i < 3 else 0  # Wrap around to connect the last point to the first
        cv2.line(frame, (int(points[i][0][0]), int(points[i][0][1])), (int(points[next_index][0][0]), int(points[next_index][0][1])), (0, 255, 0), 2)
        cv2.line(frame, (int(points[i + 4][0][0]), int(points[i + 4][0][1])), (int(points[next_index + 4][0][0]), int(points[next_index + 4][0][1])), (0, 255, 0), 2)


def create_cuboid_object(height, width, depth):
    return np.array([
        [0, 0, 0],         # Bottom center
        [width, 0, 0],     # Bottom front right
        [width, 0, -depth],  # Bottom back right
        [0, 0, -depth],    # Bottom back left
        [0, height, 0],    # Top center
        [width, height, 0],  # Top front right
        [width, height, -depth],  # Top back right
        [0, height, -depth]   # Top back left
    ], dtype=np.float32)

def translate_x(object_matrix, value):
    '''
        x-axis translation
    '''
    for i in range(len(object_matrix)):
        object_matrix[i][0] += value 
    
    return object_matrix

def translate_y(object_matrix, value):
    '''
        y-axis translation
    '''
    for i in range(len(object_matrix)):
        object_matrix[i][1] += value # x axis translation
    
    return object_matrix

def translate_z(object_matrix, value):
    '''
        z-axis translation
    '''
    for i in range(len(object_matrix)):
        object_matrix[i][2] += value # x axis translation
    
    return object_matrix

def rotate_x(object_matrix, value_in_degrees):
    rotation_angle = (value_in_degrees*2*np.pi)/360  # Adjust the angle as needed

    # Create the 3x3 rotation matrix for the X-axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [0, np.sin(rotation_angle), np.cos(rotation_angle)]])

    for i in range(len(object_matrix)):
        object_matrix[i] = np.dot(rotation_matrix, object_matrix[i])

def rotate_y(object_matrix, value_in_degrees):
    rotation_angle = (value_in_degrees*2*np.pi)/360  # Adjust the angle as needed

    # Create the 3x3 rotation matrix for the Y-axis
    rotation_matrix = np.array([[np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                                [0, 1, 0],
                                [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]])

    for i in range(len(object_matrix)):
        object_matrix[i] = np.dot(rotation_matrix, object_matrix[i])

def rotate_z(object_matrix, value_in_degrees):
    rotation_angle = (value_in_degrees*2*np.pi)/360  # Adjust the angle as needed

    # Create the 3x3 rotation matrix for the Z-xis
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1]])
    
    for i in range(len(object_matrix)):
        object_matrix[i] = np.dot(rotation_matrix, object_matrix[i])

def render_sword_object(frame, rotation_count, rvecs, tvecs, cameraMatrix, distCoeffs):

    blade_height = 50
    blade_width = 8
    blade_depth = 8
    sword_blade = create_cuboid_object(blade_height, blade_width, blade_depth)

    guard_height = 1
    guard_width = 30
    guard_depth = 16
    sword_guard = create_cuboid_object(guard_height,guard_width,guard_depth)

    pommel_height = 5
    pommel_width = 5
    pommel_depth = 5
    sword_pommel = create_cuboid_object(pommel_height,pommel_width,pommel_depth)

    if rotation_count == 0:
        translate_x(sword_blade, (14-blade_width)/2)
        translate_y(sword_blade, 14-blade_height)
        translate_z(sword_blade, blade_depth/2)

        translate_x(sword_guard, (14-guard_width)/2)
        translate_y(sword_guard, 14-guard_height)
        translate_z(sword_guard, guard_depth/2)

        translate_x(sword_pommel, (14-pommel_width)/2)
        translate_y(sword_pommel, 24+pommel_height)
        translate_z(sword_pommel, pommel_depth/2)
            
    elif rotation_count == 1: # for counterclockwise 90
        rotate_z(sword_blade, 270)
        rotate_z(sword_guard, 270)
        rotate_z(sword_pommel, 270)

        translate_y(sword_blade, (14-blade_width)/2 + blade_width)
        translate_z(sword_blade, blade_depth/2)

        translate_y(sword_guard, (14-guard_width)/2 + guard_width)
        translate_z(sword_guard, guard_depth/2)

        translate_x(sword_pommel, -10-(2*pommel_height))
        translate_y(sword_pommel, (14-pommel_width)/2 + pommel_width)
        translate_z(sword_pommel, pommel_depth/2)

    elif rotation_count == 2: # for 180
        translate_x(sword_blade, (14-blade_width)/2)
        translate_z(sword_blade, blade_depth/2)
        
        translate_x(sword_guard, (14-guard_width)/2)
        translate_z(sword_guard, guard_depth/2)

        translate_x(sword_pommel, (14-pommel_width)/2)
        translate_y(sword_pommel, -10-(2*pommel_height))
        translate_z(sword_pommel, pommel_width/2)

    else: # for clockwise 90
        rotate_z(sword_blade, 90)
        rotate_z(sword_guard, 90)
        rotate_z(sword_pommel, 90)

        translate_x(sword_blade, 14)
        translate_y(sword_blade, (14-blade_width)/2)
        translate_z(sword_blade, blade_depth/2)

        translate_x(sword_guard, 14)
        translate_y(sword_guard, (14-guard_width)/2)
        translate_z(sword_guard, guard_depth/2)

        translate_x(sword_pommel, 14+10+(2*pommel_height))
        translate_y(sword_pommel, (14-pommel_width)/2)
        translate_z(sword_pommel, pommel_width/2)

    render_cuboid(frame, sword_blade, rvecs, tvecs, cameraMatrix, distCoeffs)
    render_cuboid(frame, sword_guard, rvecs, tvecs, cameraMatrix, distCoeffs)
    render_cuboid(frame, sword_pommel, rvecs, tvecs, cameraMatrix, distCoeffs)

    return frame

def detect_marker_on_frame(frame, original_marker, cameraMatrix, distCoeffs, is_show = 0):

    binary_image = image_binarization(frame, 150, 0, 1)

    component_images = connected_components(binary_image, frame)

    component_contours = []
    for component_image in component_images:
        contours = find_contours(component_image)
        if len(contours) != 0:
            component_contours.append(contours[0])
    
    group_of_corners = find_corners_from_countours(frame, component_contours)

    detected_marker_corners = []

    rotation_count=0
    best_result = 0
    for group in group_of_corners:
        selected_corners = group[0]

        square_size =  original_marker.shape[0]  # Adjust this value as needed

        # Create the normalized square points
        normalized_corners = np.array([[0, 0], [square_size, 0], [square_size, square_size], [0, square_size]], dtype=np.float32)

        # Calculate the perspective transformation matrix
        M = cv2.getPerspectiveTransform(np.float32(selected_corners), normalized_corners)

        # Apply the perspective transformation
        normalized_square = cv2.warpPerspective(frame, M, (square_size, square_size))
        

        # Ensure that both images have the same dimensions
        normalized_square = cv2.resize(normalized_square, (original_marker.shape[1], original_marker.shape[0]))

        # Convert the images to grayscale for SSIM calculation
        original_marker_gray = cv2.cvtColor(original_marker, cv2.COLOR_BGR2GRAY)
        normalized_square_gray = cv2.cvtColor(normalized_square, cv2.COLOR_BGR2GRAY)
        normalized_square_gray = cv2.flip(normalized_square_gray, 1)
        _, normalized_square_gray = cv2.threshold(normalized_square_gray, 128, 255, cv2.THRESH_BINARY)
        normalized_square_gray = cv2.transpose(normalized_square_gray)
        normalized_square_gray = cv2.flip(normalized_square_gray, flipCode=0)

        # Compare the detected marker with the original marker image in the 4 possible rotations
        for i in range(4):
            if i == 0:
                cv2.imshow("The detected marker = ",  normalized_square_gray)
                
            # Use Template Matching to compare the detected marker with the marker image
            result = cv2.matchTemplate(original_marker_gray, normalized_square_gray, cv2.TM_CCORR_NORMED)
            
            if result > 0.95 and result > best_result:
                rotation_count = i
                detected_marker_corners = selected_corners
                detected_marker = normalized_square_gray
                best_result = result

            # Rotate 90 degrees counterclockwise
            normalized_square_gray = cv2.transpose(normalized_square_gray)
            normalized_square_gray = cv2.flip(normalized_square_gray, flipCode=0)

    
    if len(detected_marker_corners) == 0:
        return frame
    
    detected_marker_corners = np.array([detected_marker_corners[0], detected_marker_corners[3], detected_marker_corners[2], detected_marker_corners[1]])

    if is_show == 1:
        cv2.polylines(frame, [np.array(detected_marker_corners)], isClosed=True, color=(0, 255, 0), thickness=2)
        for i, point in enumerate(detected_marker_corners):
            x, y = point
            cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Draw a circle at the corner
            text = f"({x}, {y})"
            cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    rvecs, tvecs  = estimate_pose_single_marker(detected_marker_corners, 14, cameraMatrix, distCoeffs)

    frame = render_sword_object(frame, rotation_count, rvecs, tvecs, cameraMatrix, distCoeffs)

    if is_show == 1:
        cv2.imshow("Webcam Feed ", frame)
        if detected_marker is not None:
            cv2.imshow("Detected Marker Normalized ", detected_marker)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return frame


# detect_marker_on_frame(image,marker, 1)
# test(image_path=image_path, marker_path=marker_path)