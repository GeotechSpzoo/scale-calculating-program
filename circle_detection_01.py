# import the necessary packages
import numpy as np
import argparse
import cv2

def detect_circles(gray, orig):
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="Path to the image")
    # argv = ["", "-istr1/0_str1_0_max_Normal.jpg"]
    # args = vars(ap.parse_args(argv[1:]))
    # # load the image, clone it for output, and then convert it to grayscale
    # image = cv2.imread(args["image"])
    # output = image.copy()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    # cv2.HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    # ensure at least some circles were found
    # for i in range(11):
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(orig, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(orig, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        # cv2.imshow("orig", np.hstack([gray, orig]))
        cv2.imshow("orig", orig)
        cv2.waitKey(0)
