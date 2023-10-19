import cv2
import numpy as np

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

def detect_marker_on_frame(frame, original_marker):

    binary_image = image_binarization(frame, 150, 0, 1)

    _, (_, component_contours) = connected_components(binary_image, frame)

    group_of_corners, group_of_contours = find_corners_from_countours(frame, component_contours)

    detected_marker_corners = []
    detected_group_contours = []
    ix=0

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
            
            print(result)
            if result > 0.70:
                detected_marker_corners = selected_corners
                detected_group_contours = group_of_contours[ix]
                break
        ix+=1

    cv2.drawContours(frame, detected_group_contours, -1, (0, 255, 0), 5)
    for point in detected_marker_corners:
        x, y = point
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
    
    return frame, np.array(detected_marker_corners)

image_path = '../img/samples/examples/example_2.png'
marker_path = '../img/samples/marker_1.png'

# detect_marker(image_path=image_path, marker_path=marker_path)
# test(image_path=image_path, marker_path=marker_path)