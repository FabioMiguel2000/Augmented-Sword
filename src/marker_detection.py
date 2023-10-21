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

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corners[0], tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corners, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

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

    num_connected_components = 0
    new_labels = []

    # Create a list to store the images and contours of each component
    component_images = []
    component_contours = []

    for label in range(1, num_labels):

        x, y, w, h, area = stats[label]
        aspect_ratio = float(w) / h
        
        # Filter out small regions based on area and non-rectangular regions
        if stats[label, cv2.CC_STAT_AREA] >= MIN_AREA_THRESHOLD and ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX:
            
            # Generate a random color for each label
            color = np.random.randint(0, 255, size=3)
            
            # Set pixels with the current label to the chosen color
            colored_label_image[labeled_image == label] = color

            num_connected_components +=1
            new_labels.append(label)

            component_mask = (labeled_image == label).astype(np.uint8)
    
            # Apply the mask to the original image to extract the component
            component_image = cv2.bitwise_and(image, image, mask=component_mask)
            
            # Store the component image
            component_images.append(component_image)
            
            # Convert the component image to grayscale
            component_gray = cv2.cvtColor(component_image, cv2.COLOR_BGR2GRAY)
            
            # Find the contour of the component
            contours, _ = cv2.findContours(component_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Store the contour
            if len(contours) != 0:
                component_contours.append(contours[0])

    return colored_label_image, (component_images, component_contours)

def find_contours(image):
    # Apply a Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred, threshold1=30, threshold2=150)

    # Find the contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print("contour", contours)
    return contours


def find_corners_from_countours(image, contours):
    # Initialize a list to store the filtered corners
    filtered_corners = []
    group_of_corners = []
    group_of_contours = []

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
                group_of_contours.append([contour])
                filtered_corners.extend(approx)

    corner_image = image.copy()

    for point in filtered_corners:
        x, y = point
        cv2.circle(corner_image, (x, y), 20, (255, 0, 0), -1)
    
    # cv2.imshow('Contour Detection with Corners', corner_image)

    print("Relevant Connected Components found = ", len(group_of_corners))

    return group_of_corners, group_of_contours

def detect_marker(frame = object, image_path= '', marker_path = ''):

    if image_path == '':
        image = frame
    else:    
        image = cv2.imread(image_path)

    original_marker = cv2.imread(marker_path)

    binary_image = image_binarization(image, 150, 0, 1)

    colored_label_image = connected_components(binary_image)

    # Display the labeled image
    cv2.imshow('Connected Component Image', colored_label_image)

    contours = find_contours(colored_label_image)

    # Draw the contours on the original image
    contour_image = image.copy()

    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Display the image with detected contours
    # cv2.imshow('Contour Detection', contour_image)

    group_of_corners, group_of_contours = find_corners_from_countours(image, contours)

    best_result = 0 

    detected_marker_corners = []
    i=0

    for group in group_of_corners:
        selected_corners = group[0]

        selected_corners = sorted(selected_corners, key=lambda x: x[0])
        selected_corners = sorted(selected_corners, key=lambda x: x[1])

        square_size =  original_marker.shape[0]  # Adjust this value as needed

        # Create the normalized square points
        normalized_corners = np.array([[0, 0], [square_size, 0], [0, square_size], [square_size, square_size]], dtype=np.float32)

        # Calculate the perspective transformation matrix
        M = cv2.getPerspectiveTransform(np.float32(selected_corners), normalized_corners)

        # Apply the perspective transformation
        normalized_square = cv2.warpPerspective(image, M, (square_size, square_size))
        
        # cv2.imshow("Original Marker ", original_marker)

        # Ensure that both images have the same dimensions
        normalized_square = cv2.resize(normalized_square, (original_marker.shape[1], original_marker.shape[0]))

        # Convert the images to grayscale for SSIM calculation
        original_marker_gray = cv2.cvtColor(original_marker, cv2.COLOR_BGR2GRAY)
        normalized_square_gray = cv2.cvtColor(normalized_square, cv2.COLOR_BGR2GRAY)
        normalized_square_gray = cv2.flip(normalized_square_gray, 1)
        _, normalized_square_gray = cv2.threshold(normalized_square_gray, 128, 255, cv2.THRESH_BINARY)
        # Compare the detected marker with the original marker image in the 4 possible rotations
        for i in range(4):
            # Rotate 90 degrees counterclockwise
            normalized_square_gray = cv2.transpose(normalized_square_gray)
            normalized_square_gray = cv2.flip(normalized_square_gray, flipCode=0)
            # mse = ((rotated_image - image2_gray) ** 2).mean() # Sum square difference
            result = cv2.matchTemplate(original_marker_gray, normalized_square_gray, cv2.TM_CCORR_NORMED)
            
            if best_result < result:
                detected_marker_corners = selected_corners
                detected_marker = normalized_square_gray
                best_result = result
        i+=1

    print("Template Matching Result = ", best_result)

    cv2.imshow("Detected Marker", detected_marker)

    # cv2.imshow("Detected Marker", detected_marker)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if detected_marker_corners == [] or best_result[0][0] < 0.7:
        return [], []
    return detected_marker_corners, group_of_contours

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    # marker_points = np.array([[marker_size / 2, -marker_size / 2, 0],
    #                         [marker_size / 2, marker_size / 2, 0],
    #                         [-marker_size / 2, marker_size / 2, 0],
    #                         [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    corners = np.array(corners, dtype=np.float32)
    print("here >>>>>>>", corners)
    trash = []
    rvecs = []
    tvecs = []
    i = 0
    # for c in corners:
    #     print("?>>>>>>>", corners[i])
    nada, R, t = cv2.solvePnP(marker_points, corners, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    rvecs.append(R)
    tvecs.append(t)
    trash.append(nada)

    # for c in corners:
    #     nada, R, t = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    #     rvecs.append(R)
    #     tvecs.append(t)
    #     trash.append(nada)
    return rvecs, tvecs, trash

def detect_marker_on_frame(frame, original_marker, is_show = 0):

    binary_image = image_binarization(frame, 150, 0, 1)

    _, (_, component_contours) = connected_components(binary_image, frame)

    group_of_corners, _ = find_corners_from_countours(frame, component_contours)

    detected_marker_corners = []
    # detected_group_contours = []
    ix=0

    for group in group_of_corners:
        selected_corners = group[0]

        print("HERE : ", selected_corners)

        # selected_corners = sorted(selected_corners, key=lambda x: x[0])
        # selected_corners = sorted(selected_corners, key=lambda x: x[1])

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
        # Compare the detected marker with the original marker image in the 4 possible rotations
        for i in range(4):
            # Rotate 90 degrees counterclockwise
            normalized_square_gray = cv2.transpose(normalized_square_gray)
            normalized_square_gray = cv2.flip(normalized_square_gray, flipCode=0)

            # Use Template Matching to compare the detected marker with the marker image
            result = cv2.matchTemplate(original_marker_gray, normalized_square_gray, cv2.TM_CCORR_NORMED)
            
            # print(result)
            if result > 0.90:
                print("entered")
                detected_marker_corners = selected_corners
                # detected_group_contours = group_of_contours[ix]
                
                detected_marker = normalized_square_gray
                break
        ix+=1
    detected_marker_corners = sorted(selected_corners, key=lambda point: point[0])

    # Determine the left and right corners
    left_corners = detected_marker_corners[:2]
    right_corners = detected_marker_corners[2:]

    # Sort the left and right corners by their y-coordinates
    left_corners = sorted(left_corners, key=lambda point: point[1])
    right_corners = sorted(right_corners, key=lambda point: point[1])

    # Separate the corners as top-left, top-right, bottom-right, and bottom-left
    top_left, top_right, bottom_right, bottom_left = left_corners[0], right_corners[0], right_corners[1], left_corners[1]

    detected_marker_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    # cv2.drawContours(frame, detected_group_contours, -1, (0, 255, 0), 5)
    cv2.polylines(frame, [np.array(detected_marker_corners)], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, point in enumerate(detected_marker_corners):
        x, y = point
        cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Draw a circle at the corner
        text = f"({x}, {y})"
        print("Points detected:", point)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    if is_show == 1:
        cv2.imshow("Webcam Feed ", frame)
        if detected_marker is not None:
            cv2.imshow("Detected Marker Normalized ", detected_marker)

            rvecs, tvecs, trash = my_estimatePoseSingleMarkers(detected_marker_corners, 14, cameraMatrix, distCoeffs)
            # objp = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)

            # ret,rvecs, tvecs = cv2.solvePnP(objp, detected_marker_corners, cameraMatrix, distCoeffs)
            rvecs = np.array(rvecs, dtype=np.float32)
            tvecs = np.array(tvecs, dtype=np.float32)
            print(">>>>RVECS:\n", rvecs)
            print(">>>>TVECS:\n", tvecs)
            print(">>>>Trash:\n", trash)
             # project 3D points to image plane

            # axis = np.float32([[10,0,0], [0,10,0], [0,0,10]]).reshape(-1,3)
            axis_points_3D = np.array([
                [0, 0, 0],                     # Origin
                [7, 0, 0],           # X-axis endpoint
                [0, 7, 0],           # Y-axis endpoint
                [0, 0, 7]            # Z-axis endpoint
            ], dtype=np.float32)

            axis_points_2D, jac = cv2.projectPoints(axis_points_3D, rvecs, tvecs, cameraMatrix, distCoeffs)

            axis_points_2D = np.int32(axis_points_2D).reshape(-1, 2)

            # Draw X, Y, and Z axes on the image
            frame = cv2.line(frame, tuple(axis_points_2D[0]), tuple(axis_points_2D[1]), (0, 0, 255), 5)  # Red X-axis
            frame = cv2.line(frame, tuple(axis_points_2D[0]), tuple(axis_points_2D[2]), (0, 255, 0), 5)  # Green Y-axis
            frame = cv2.line(frame, tuple(axis_points_2D[0]), tuple(axis_points_2D[3]), (255, 0, 0), 5)  # Blue Z-axis

            # Display or save the image with the axes
            cv2.imshow("Image with Axes", frame)
            # frame = draw(frame,detected_marker_corners,imgpts)
            # cv2.imshow('img',frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return frame, np.array(detected_marker_corners )


detect_marker_on_frame(image,marker, 1)
# test(image_path=image_path, marker_path=marker_path)