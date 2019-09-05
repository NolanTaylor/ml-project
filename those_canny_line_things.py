import cv2
import pandas

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "C:\PythonPrograms\mlproject\Images\opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

        imgFile = cv2.imread(img_name)
        gray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        im_bw = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(im_bw, contours, -1, (0,255,0), 1)
        mask = cv2.inRange(im_bw, (0, 200, 0), (20, 255, 20))
        cv2.imshow('dst_rt', mask)
        cv2.imwrite(img_name, mask)
        cv2.waitKey(1)

cam.release()

cv2.destroyAllWindows()