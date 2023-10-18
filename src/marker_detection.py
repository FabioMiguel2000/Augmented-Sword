import cv2
import numpy as np

image = cv2.imread('./example2.png')
original_marker = cv2.imread('marker.png')

# Define a function to calculate the angle between two vectors
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    return np.arccos(dot_product / (magnitude_v1 * magnitude_v2))

def regionLabelling():
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Binarization (Thresholding)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    inverted_image = cv2.bitwise_not(binary)

    cv2.imshow('Binary', inverted_image)

    num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(inverted_image, connectivity=8)

    # `num_labels` gives the total number of labeled regions
    # `labeled_image` is an image with each pixel labeled with its region's ID
    # `stats` is a NumPy array containing statistics for each labeled region
    # `centroids` contains the centroids of each labeled region
    # cv2.imshow('Labeled', labeled_image)

    # Create a colored label image
    colored_label_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)

    min_area_threshold = 20000  # Adjust this threshold as needed

    # Define the aspect ratio range for rectangular regions
    aspect_ratio_min = 0.7
    aspect_ratio_max = 1.45

    label_num = 0

    for label in range(1, num_labels):

        x, y, w, h, area = stats[label]
        aspect_ratio = float(w) / h
        
        # Filter out small regions based on area and non-rectangular regions
        if stats[label, cv2.CC_STAT_AREA] >= min_area_threshold and aspect_ratio_min <= aspect_ratio <= aspect_ratio_max:
            # Generate a random color for each label
            color = np.random.randint(0, 255, size=3)
            
            # Set pixels with the current label to the chosen color
            colored_label_image[labeled_image == label] = color

            label_num +=1

    print("\nConnected Components found: ", label_num)
    # Display the labeled image
    cv2.imshow('Labeled Image', colored_label_image)

    # Apply a Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(colored_label_image, (5, 5), 0)

    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred, threshold1=30, threshold2=150)

    # Find the contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    contour_image = image.copy()

    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Display the image with detected contours
    cv2.imshow('Contour Detection', contour_image)

    # Initialize a list to store the filtered corners
    filtered_corners = []
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
                filtered_corners.extend(approx)

    # Draw the filtered corners on a copy of the original image
    corner_image = image.copy()
    for point in filtered_corners:
        x, y = point
        cv2.circle(corner_image, (x, y), 20, (255, 0, 0), -1)
    
    cv2.imshow('Contour Detection with Corners', corner_image)

    best_result = 0 

    print("Countours found = ", len(group_of_corners))

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
        
        cv2.imshow("Original Marker ", original_marker)

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
                detected_marker = normalized_square_gray
                best_result = result

    print(best_result)

    cv2.imshow("Detected Marker", detected_marker)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

regionLabelling()