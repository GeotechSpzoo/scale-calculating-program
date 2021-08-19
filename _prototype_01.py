import cv2

import functions as f

img = f.load_image("str1/0_str1_0_max_Normal.jpg")
gray = f.bgr_to_gray(img)
blurred = f.blur_bilateral_filter_min(gray)
for i in range(100):
    blurred = f.blur_bilateral_filter_min(blurred)
cv2.waitKey(0)
f.detect_circles(blurred, img)
cv2.waitKey(0)
