import cv2
import numpy as np

image = cv2.imread('./example2.png')

def findSquare():

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum area threshold to discard small polygons
    min_area_threshold = 20000  # Adjust this value according to your needs

    detected_markers = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the polygon has 3 to 6 vertices, it's considered a marker
        if 3 <= len(approx) <= 6 and area >= min_area_threshold:
            detected_markers.append(approx)
            print(area)

    cv2.drawContours(image, detected_markers, -1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Detected Markers', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def findUsingBlobDectetion():
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Binarization (Thresholding)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Step 3: Connected Components (Blobs) Detection
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Step 4: Checking for Blob Size
    filtered_blobs = []

    for i, stat in enumerate(stats):
        x, y, w, h, area = stat
        if area > 100 and area < 5000:  # Adjust these values based on your marker size
            filtered_blobs.append(i)
            print("filtered blob = ", i)

    # Step 5: Detect Borders of Blobs
    border_image = np.zeros_like(binary)
    for i in filtered_blobs:
        blob_mask = np.uint8(labels == i)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        cv2.drawContours(border_image, contours, -1, 255, -1)

    # Step 6: Detect Corners of Blobs
    corner_image = np.zeros_like(binary)
    corners = cv2.goodFeaturesToTrack(border_image, maxCorners=4, qualityLevel=0.01, minDistance=10)
    corners = np.int0(corners)

    print(corners)

    # Step 7: Reject Blobs That Are Not Quadrilaterals
    final_markers = []

    for i in filtered_blobs:
        if len(corners) == 4:
            final_markers.append(i)
            print("final")

    # Step 8: Refine Corner Coordinates (if needed)
    # You can use a corner refinement technique like sub-pixel corner detection here if desired.

    # Draw the final markers on the original image
    for i in final_markers:
        blob_mask = np.uint8(labels == i)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Detected Markers', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def regionLabelling():
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Binarization (Thresholding)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

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

    min_area_threshold = 1000  # Adjust this threshold as needed

    # Define the aspect ratio range for rectangular regions
    aspect_ratio_min = 0.7
    aspect_ratio_max = 1.45

    label_num = 0

    for label in range(1, num_labels):

        x, y, w, h, area = stats[label]
        aspect_ratio = float(w) / h
        
        # Filter out non-rectangular regions
        if stats[label, cv2.CC_STAT_AREA] >= min_area_threshold and aspect_ratio_min <= aspect_ratio <= aspect_ratio_max:
            # Generate a random color for each label
            color = np.random.randint(0, 255, size=3)
            
            # Set pixels with the current label to the chosen color
            colored_label_image[labeled_image == label] = color

    print("Regions found", label_num)
    # Display the labeled image
    cv2.imshow('Labeled Image', colored_label_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

regionLabelling()