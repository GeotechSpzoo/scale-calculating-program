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


REF_OBJ_SIZE_IN_INCH = 10.0
PIXELS_PER_METRIC = None

# STEP1 - Read image and define pixel size
img = cv2.imread("11-11_o1_0_min_UV.jpg", cv2.IMREAD_COLOR)
# cv2.imshow("img", img)
# cv2.waitKey(0)
contrast = increase_contrast(img)
cv2.imshow("contrast0", contrast)
cv2.waitKey(0)
# contrast1 = increase_contrast(contrast0.copy())
# cv2.imshow("contrast1", contrast1)
# cv2.waitKey(0)
# contrast2 = increase_contrast(contrast1.copy())
# cv2.imshow("contrast2", contrast2)
# cv2.waitKey(0)

gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

# Step 2: Denoising, if required and threshold image

# No need for any denoising or smoothing as the image looks good.
# Otherwise, try Median or NLM
# plt.hist(img.flat, bins=100, range=(0,255))

# Change the grey image to binary by thresholding.
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# print(ret)  #Gives 157 on grains2.jpg. OTSU determined this to be the best threshold.

# View the thresh image. Some boundaries are ambiguous / faint.
# Some pixles in the middle.
# Need to perform morphological operations to enhance.
cv2.imshow("thershold", thresh)
cv2.waitKey(0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(thresh, 27, 369)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

edged = cv2.dilate(edged, None, iterations=1)
cv2.imshow("Dilated", edged)
cv2.waitKey(0)
edged = cv2.erode(edged, None, iterations=1)
cv2.imshow("Eroded", edged)
cv2.waitKey(0)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)

# loop over the contours individually
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 1:
        continue

    # compute the rotated bounding box of the contour
    orig = img.copy()
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
    cv2.putText(orig, "{:.1f}in".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1)
    cv2.putText(orig, "{:.1f}in".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1)
    # show the output image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
