import numpy as np
import cv2 as cv
import glob

CHESS_BOARD_ROWS=6
CHESS_BOARD_COLS=8

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESS_BOARD_ROWS*CHESS_BOARD_COLS,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESS_BOARD_ROWS,0:CHESS_BOARD_COLS].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('../../img/samples/training/*.png')

print(images)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (CHESS_BOARD_ROWS,CHESS_BOARD_COLS), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        cv.drawChessboardCorners(img, (CHESS_BOARD_ROWS,CHESS_BOARD_COLS), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
print("\n\nCamera Calibrated:\n", ret)
print("\n\nCamera Matrix:\nfx:\t",mtx[0][0],
      "\nfy:\t", mtx[1][1],
      "\ncx:\t", mtx[0][2],
      "\ncy:\t", mtx[1][2])

print("\n\nDistortion Coefficients:\nk1x:\t",dist[0][0],
      "\nk2:\t", dist[0][1],
      "\np1:\t", dist[0][2],
      "\np2:\t", dist[0][3],
      "\nk3:\t", dist[0][4])
print("\n\nRotation Vectors:\n ", rvecs)
print("\n\nTranslation Vectors:\n ", tvecs)

f = open("../constants.py", "w")
f.write("fx="+ str(mtx[0][0]) +
      "\nfy="+ str(mtx[1][1]) +
      "\ncx="+ str(mtx[0][2])+
      "\ncy="+ str(mtx[1][2])+
      "\nk1=" + str(dist[0][0]) +
      "\nk2="+ str(dist[0][1]) +
      "\np1="+ str(dist[0][2]) +
      "\np2="+ str(dist[0][3]) +
      "\nk3="+ str(dist[0][4]) +
      "\nmtx=" + str(mtx) +
      "\ndist=" + str(dist)     
      )
f.close()

img = cv.imread('../../img/samples/training/uncalibrated_image1.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('../../img/samples/results/calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "\n\ntotal error: {}".format(mean_error/len(objpoints)) )

