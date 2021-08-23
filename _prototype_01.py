import cv2

import functions as f

img = f.load_image("str1/0_str1_0_max_Normal.jpg")
# contrasted = f.contrast_increase_clahe(img)
# for i in range(1):
#     contrasted = f.contrast_increase_clahe(contrasted)
gray = f.bgr_to_gray(img)
blurred = f.blur_bilateral_filter_min(gray)
for i in range(10):
    blurred = f.blur_bilateral_filter_min(blurred)
cv2.waitKey(0)
# for i in range(255):
#     binary = f.gray_to_binary(blurred, i)
# cv2.waitKey(0)
binary = f.gray_to_binary(blurred, 77)
edged = f.detect_edges_raw_canny(binary, 25, 100)
eroded = f.erode_dilate(edged)
f.find_contours_and_draw_them(f.gray_to_bgr(gray), eroded, "window name", 72)
cv2.waitKey(0)
