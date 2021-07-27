# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

KEY_ESC = 27
KEY_SPACE = 32
KEY_BACKSPACE = 8


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


def find_contours_and_draw_them(img, edged, window_name):
    global PIXELS_PER_METRIC
    # find contours in the edge map
    # cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
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
    cv2.imshow(window_name, orig)
    cv2.waitKey(0)


PHOTO_PATH = "piasekSredni/0_piasekSredni_0_min_UV.jpg"
REF_OBJ_SIZE_IN_INCH = 10.0
PIXELS_PER_METRIC = None

# STEP1 - Read image and define pixel size
img = cv2.imread(PHOTO_PATH, cv2.IMREAD_COLOR)
cv2.imshow("img", img)
cv2.waitKey(0)
# # Sharpen image
# kernel = np.array([[-1.1, -1.1, -1.1],
#                    [-1.1, 9.8, -1.1],
#                    [-1.1, -1.1, -1.1]])
# sharpened = cv2.filter2D(img, -1, kernel)  # applying the sharpening kernel to the input image & displaying it.
# cv2.imshow('Sharpened', sharpened)
# cv2.waitKey(0)
# cv2.imshow("contrast0", img)
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

# Find edges using canny edge detector
def auto_canny(grayim, sigma=0.33, v=202):
    # compute the median of the single channel pixel intensities
    # v = np.median(grayim)  # 202.0
    # v = np.float64(20)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(grayim, lower, upper)
    # return the edged image
    return edged


# Find edges using canny edge detector
def auto_canny_default(grayim):
    # compute the median of the single channel pixel intensities
    # v = np.median(grayim)  # 202.0
    sigma = 0.33
    v = np.float64(20)
    print(f"Before Canny Median {v}")
    print(f"V is {type(v)}")
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(grayim, lower, upper)
    # return the edged image
    return edged


d = 4
sigmaParam = 1
cannySigma = 0.11
cannyV = 30
blurred = gray.copy()
temp = cv2.bilateralFilter(gray.copy(), d, sigmaParam, sigmaParam)
edged = auto_canny_default(temp)
blrCount = 0
for n in range(5001):
    temp = cv2.bilateralFilter(temp.copy(), d, sigmaParam, sigmaParam)
    if n % 25 == 0:
        cv2.imshow(f"blur {n}", temp)
        cv2.waitKey(0)
        # contrast = increase_contrast(temp)
        # cv2.imshow(f"Contrast {n}", contrast)
        # cv2.waitKey(0)
        # # Sharpen image
        # kernel = np.array([[-1.0, -1.1, -1.0],
        #                    [-1.0, 8.5, -1.0],
        #                    [-1.0, -1.0, -1.0]])
        # sharpened = cv2.filter2D(temp, -1, kernel)  # applying the sharpening kernel to the input image & displaying it.
        # cv2.imshow(f"Sharpened {n}", sharpened)
        # cv2.waitKey(0)
        key = None
        for i in range(20):
            for j in range(25):
                edged = auto_canny(temp.copy(), i / 10, j * 10)
                cv2.imshow(f"Blur {blrCount}, auto_canny i: {i} j: {j}           sigma: [{i / 10}] v: [{j * 10}]",
                           edged)
                key = cv2.waitKey(0)
                edged = cv2.dilate(edged, None, iterations=1)
                cv2.imshow(f"Dilated {blrCount}, auto_canny i: {i} j: {j}           sigma: [{i / 10}] v: [{j * 10}]",
                           edged)
                cv2.waitKey(0)
                edged = cv2.erode(edged, None, iterations=1)
                cv2.imshow(f"Eroded {blrCount}, auto_canny i: {i} j: {j}           sigma: [{i / 10}] v: [{j * 10}]",
                           edged)
                cv2.waitKey(0)
                find_contours_and_draw_them(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR), edged,
                                            f"Contours {blrCount}, auto_canny i: {i} j: {j}           sigma: [{i / 10}] v: [{j * 10}]")
                if key == KEY_ESC:
                    break
                elif key == KEY_BACKSPACE:
                    break
                else:
                    continue
            if key == KEY_ESC:
                break
            elif key == KEY_BACKSPACE:
                continue
            else:
                continue
        blrCount += 1

# blurs = [temp]
# # perform edge detection, then perform a dilation + erosion to
# # close gaps in between object
# blrCount = 0
# for blr in blurs:
#     edged = blr.copy()
#     edged = auto_canny_default(edged)
#     cv2.imshow(f"Blur {blrCount}, auto_canny default params", edged)
#     cv2.waitKey(0)
#     key = None
#     for i in range(20):
#         for j in range(25):
#             edged = auto_canny(blr, i / 10, j * 10)
#             cv2.imshow(f"Blur {blrCount}, auto_canny i: {i} j: {j}           sigma: [{i/10}] v: [{j*10}]", edged)
#             key = cv2.waitKey(0)
#             if key == KEY_ESC:
#                 break
#             elif key == KEY_BACKSPACE:
#                 break
#             else:
#                 continue
#         if key == KEY_ESC:
#             break
#         elif key == KEY_BACKSPACE:
#             continue
#         else:
#             continue
#     blrCount += 1
# blrCount = 0
# for blr in blurs:
#     edged = blr.copy()
#     key = None
#     blrCount += 1
#     for i in range(21):
#         for j in range(21):
#             edged = cv2.Canny(blr, 20*i, 20*j, edged, L2gradient=cv2.NORM_L2)
#             cv2.imshow(f"Blur {blrCount}, Edged i: {i} j: {j}           threshold1: [{20*i}] threshold2: [{20*j}]", edged)
#             key = cv2.waitKey(0)
#             if key == KEY_ESC:
#                 break
#             elif key == KEY_BACKSPACE:
#                 break
#             else:
#                 continue
#         if key == KEY_ESC:
#             break
#         elif key == KEY_BACKSPACE:
#             continue
#         else:
#             continue
# edged = cv2.Canny(gray, 60, 200, L2gradient=cv2.NORM_L2)
# cv2.imshow("Edged2", edged)
# cv2.waitKey(0)
# edged = cv2.Canny(gray, 60, 255, L2gradient=cv2.NORM_L2)
# cv2.imshow("Edged3", edged)
# cv2.waitKey(0)
# edged = cv2.Canny(gray, 60, 2048, L2gradient=cv2.NORM_L2)
# cv2.imshow("Edged4", edged)
# cv2.waitKey(0)

# edged = cv2.dilate(edged, None, iterations=1)
# cv2.imshow("Dilated", edged)
# cv2.waitKey(0)
# edged = cv2.erode(edged, None, iterations=1)
# cv2.imshow("Eroded", edged)
# cv2.waitKey(0)
