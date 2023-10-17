import cv2

image = cv2.imread('./example2.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray', gray)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.imshow('blurred', blurred)

edges = cv2.Canny(blurred, 50, 150)

cv2.imshow('edges', edges)

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