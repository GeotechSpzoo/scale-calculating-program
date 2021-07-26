# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def increase_contrast(imgToContrast):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab", lab)
    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)
    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(128, 128))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)
    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    # cv2.imshow('limg', limg)
    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cv2.imshow('final', final)
    # _____END_____#
    return final


PHOTO_PATH = "piasekSredni/0_piasekSredni_0_min_Normal.jpg"
REF_OBJ_SIZE_IN_INCH = 10.0
PIXELS_PER_METRIC = None

# STEP1 - Read image and define pixel size
img = cv2.imread(PHOTO_PATH, cv2.IMREAD_COLOR)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# # Sharpen image
# kernel = np.array([[-1.1, -1.1, -1.1],
#                    [-1.1, 9.8, -1.1],
#                    [-1.1, -1.1, -1.1]])
# sharpened = cv2.filter2D(img, -1, kernel)  # applying the sharpening kernel to the input image & displaying it.
# cv2.imshow('Sharpened', sharpened)
# cv2.waitKey(0)
# cv2.imshow("contrast0", increase_contrast(sharpened))
# cv2.waitKey(0)
# contrast1 = increase_contrast(contrast0.copy())
# cv2.imshow("contrast1", contrast1)
# cv2.waitKey(0)
# contrast2 = increase_contrast(contrast1.copy())
# cv2.imshow("contrast2", contrast2)
# cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Denoising, if required and threshold image

# No need for any denoising or smoothing as the image looks good.
# Otherwise, try Median or NLM
# plt.hist(img.flat, bins=100, range=(0,255))

# Change the grey image to binary by thresholding.
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# print(ret)  #Gives 157 on grains2.jpg. OTSU determined this to be the best threshold.

# View the thresh image. Some boundaries are ambiguous / faint.
# Some pixles in the middle.
# Need to perform morphological operations to enhance.
# cv2.imshow("thershold", thresh)
# cv2.waitKey(0)

# Blur to remove noise
blur = cv2.bilateralFilter(gray.copy(), 5, 5, 5)
cv2.imshow("blur1", blur)
cv2.waitKey(0)
blur2 = cv2.bilateralFilter(blur.copy(), 5, 5, 5)
cv2.imshow("blur2", blur2)
cv2.waitKey(0)
blur3 = cv2.bilateralFilter(blur2.copy(), 5, 5, 5)
cv2.imshow("blur3", blur3)
cv2.waitKey(0)
blur4 = cv2.bilateralFilter(blur3.copy(), 5, 5, 5)
cv2.imshow("blur4", blur4)
cv2.waitKey(0)
blur5 = cv2.bilateralFilter(blur4.copy(), 5, 5, 5)
cv2.imshow("blur5", blur5)
cv2.waitKey(0)
blur6 = cv2.bilateralFilter(blur5.copy(), 5, 5, 5)
cv2.imshow("blur6", blur6)
cv2.waitKey(0)
blur7 = cv2.bilateralFilter(blur6.copy(), 5, 5, 5)
cv2.imshow("blur7", blur7)
cv2.waitKey(0)
blur8 = cv2.bilateralFilter(blur7.copy(), 5, 5, 5)
cv2.imshow("blur8", blur8)
cv2.waitKey(0)
blur9 = cv2.bilateralFilter(blur8.copy(), 5, 5, 5)
cv2.imshow("blur9", blur9)
cv2.waitKey(0)
blur10 = cv2.bilateralFilter(blur9.copy(), 5, 5, 5)
cv2.imshow("blur10", blur10)
cv2.waitKey(0)

# Find edges using canny edge detector
def auto_canny(grayim, sigma=0.33):
    # compute the median of the single channel pixel intensities
    # v = np.median(grayim)  # 202.0
    v = np.float64(20)
    print(f"Before Canny Median {v}")
    print(f"V is {type(v)}")
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(grayim, lower, upper)
    # return the edged image
    return edged


canned = auto_canny(blur)
cv2.imshow("canned1", canned)
cv2.waitKey(0)
canned2 = auto_canny(blur2)
cv2.imshow("canned2", canned2)
cv2.waitKey(0)
canned3 = auto_canny(blur3)
cv2.imshow("canned3", canned3)
cv2.waitKey(0)
canned4 = auto_canny(blur4)
cv2.imshow("canned4", canned4)
cv2.waitKey(0)
canned5 = auto_canny(blur5)
cv2.imshow("canned5", canned5)
cv2.waitKey(0)
canned6 = auto_canny(blur6)
cv2.imshow("canned6", canned6)
cv2.waitKey(0)
canned7 = auto_canny(blur7)
cv2.imshow("canned7", canned7)
cv2.waitKey(0)
canned8 = auto_canny(blur8)
cv2.imshow("canned8", canned8)
cv2.waitKey(0)
canned9 = auto_canny(blur9)
cv2.imshow("canned9", canned9)
cv2.waitKey(0)
canned10 = auto_canny(blur10)
cv2.imshow("canned10", canned10)
cv2.waitKey(0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object
edged = blur.copy()
for i in range(21):
    for j in range(21):
        edged = cv2.Canny(gray, 20*i, 20*j, edged, L2gradient=cv2.NORM_L2)
        cv2.imshow(f"Edged i: {i} j: {j}           threshold1: [{20*i}] threshold2: [{20*j}]", edged)
        cv2.waitKey(0)
# edged = cv2.Canny(gray, 60, 200, L2gradient=cv2.NORM_L2)
# cv2.imshow("Edged2", edged)
# cv2.waitKey(0)
# edged = cv2.Canny(gray, 60, 255, L2gradient=cv2.NORM_L2)
# cv2.imshow("Edged3", edged)
# cv2.waitKey(0)
# edged = cv2.Canny(gray, 60, 2048, L2gradient=cv2.NORM_L2)
# cv2.imshow("Edged4", edged)
# cv2.waitKey(0)

edged = cv2.dilate(edged, None, iterations=1)
# cv2.imshow("Dilated", edged)
# cv2.waitKey(0)
edged = cv2.erode(edged, None, iterations=1)
# cv2.imshow("Eroded", edged)
# cv2.waitKey(0)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)

# loop over the contours individually
orig = img.copy()
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 25:
        continue

    # compute the rotated bounding box of the contour
    # orig = cv2.cvtColor(edged.copy(), cv2.COLOR_GRAY2BGR)
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 1, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    # cv2.circle(orig, (int(tltrX), int(tltrY)), 1, (255, 0, 0), -1)
    # cv2.circle(orig, (int(blbrX), int(blbrY)), 1, (255, 0, 0), -1)
    # cv2.circle(orig, (int(tlblX), int(tlblY)), 1, (255, 0, 0), -1)
    # cv2.circle(orig, (int(trbrX), int(trbrY)), 1, (255, 0, 0), -1)

    # draw lines between the midpoints
    # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
    #          (255, 0, 255), 1)
    # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
    #          (255, 0, 255), 1)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if PIXELS_PER_METRIC is None:
        PIXELS_PER_METRIC = dB / REF_OBJ_SIZE_IN_INCH
    # compute the size of the object
    dimA = dA / PIXELS_PER_METRIC
    dimB = dB / PIXELS_PER_METRIC
    # draw the object sizes on the image
    # cv2.putText(orig, "{:.1f}in".format(dimA),
    #             (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.65, (255, 255, 255), 1)
    # cv2.putText(orig, "{:.1f}in".format(dimB),
    #             (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.65, (255, 255, 255), 1)
    # show the output image
cv2.imshow("Image", orig)
cv2.waitKey(0)
