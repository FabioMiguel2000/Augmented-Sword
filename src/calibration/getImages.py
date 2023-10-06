import cv2
import os

img_dir = '../../img/samples/training/'

def delete_files_in_directory(directory_path):
   try:
     with os.scandir(directory_path) as entries:
       for entry in entries:
         if entry.is_file():
            os.unlink(entry.path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def takePictures():
    num = 1
    cameraCapture = cv2.VideoCapture(0)
    delete_files_in_directory(img_dir)

    while True:

        success, img = cameraCapture.read()

        k = cv2.waitKey(5)

        if k == ord('s'):
            cv2.imwrite(img_dir + 'uncalibrated_image'+ str(num) +'.png', img)
            print('Image was successfully saved!')
            num += 1
        if k == ord('q'):
            break

        cv2.imshow('Img', img)

    cameraCapture.release()

    cv2.destroyAllWindows()

takePictures()