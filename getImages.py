import cv2

cameraCapture = cv2.VideoCapture(1)
num = 1

while True:
    success, img = cameraCapture.read()

    k = cv2.waitKey(5)

    if k == ord('s'):
        cv2.imwrite('./img/samples/uncalibrated_image'+ str(num) +'.png', img)
        print('Image was successfully saved!')
        num += 1
    if k == ord('q'):
        break

    cv2.imshow('Img', img)

cameraCapture.release()

cv2.destroyAllWindows()
