from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

KEY_ESC = 27
KEY_SPACE = 32
KEY_BACKSPACE = 8

PHOTO_PATH = "str2/0_str2_0_max_Normal.jpg"
ONE_INCH_IN_MILLIMETERS = 25.4
ONE_MILLIMETER_IN_INCHES = 1 / 25.4
REF_OBJ_SIZE_IN_MILLIMETERS = 0.1  # 0.1 mm = 0.003937 inch
REF_OBJ_SIZE_IN_INCH = ONE_MILLIMETER_IN_INCHES * REF_OBJ_SIZE_IN_MILLIMETERS  # 0.1 mm = 0.003937 inch
PIXELS_PER_METRIC = None


def increase_contrast_clahe(img):
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
    cv2.imshow("increase_contrast_clahe", final)
    cv2.waitKey(0)
    # _____END_____#
    return final


def sharpen(img):
    # Sharpen image
    kernel = np.array([[-1.1, -1.1, -1.1],
                       [-1.1, 9.8, -1.1],
                       [-1.1, -1.1, -1.1]])
    sharpened = cv2.filter2D(img, -1, kernel)  # applying the sharpening kernel to the input image & displaying it.
    cv2.imshow("sharpen", sharpened)
    cv2.waitKey(0)
    return sharpened


def gray_to_binary(gray):
    # Change the grey image to binary by thresholding.
    thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("gray_to_binary", thresholded)
    cv2.waitKey(0)
    return thresholded


def blur_gaussian(gray):
    k_size = (7, 7)
    sigma = 0
    # GaussianBlur(src, k_size, sigmaX, dst=None, sigmaY=None, borderType=None)
    blurred = cv2.GaussianBlur(gray, k_size, sigma)
    cv2.imshow(f"blur_gaussian k_size: {k_size}, sigma: {sigma}", gray)
    cv2.waitKey(0)
    return blurred


def blur_bilateral_filter(gray):
    d = 4
    sigma = 1
    blurred = cv2.bilateralFilter(gray.copy(), d, sigma, sigma)
    cv2.imshow(f"blur_bilateral_filter d: {d}, sigma: {sigma}", gray)
    cv2.waitKey(0)
    return blurred


def load_image(photo_path):
    loaded = cv2.imread(PHOTO_PATH, cv2.IMREAD_COLOR)
    cv2.imshow(f"load_image path:{photo_path}", loaded)
    cv2.waitKey(0)
    return loaded


def rgb_to_gray(img):
    pass


def gray_to_rgb(img):
    pass


# Find edges using canny edge detector
def detect_edges_sigma_v(gray_img, sigma=0.33, v=202):
    # compute the median of the single channel pixel intensities
    # v = np.median(grayim)  # 202.0
    # v = np.float64(20)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(gray_img, lower, upper)
    cv2.imshow(f"detect_edges_sigma_v sigma={sigma}, v={v}", edged)
    cv2.waitKey(0)
    return edged


# Find edges using canny edge detector
def detect_edges_auto(gray_img, sigma=0.33, v=202):
    # compute the median of the single channel pixel intensities
    v = np.median(gray_img)  # 202.0
    # v = np.float64(20)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(gray_img, lower, upper)
    cv2.imshow(f"detect_edges_auto sigma={sigma}, v={v}, lower={lower}, upper={upper}", edged)
    cv2.waitKey(0)
    return edged


# Find edges using canny edge detector
def detect_edges_raw_canny(gray_img, lower, upper):
    edged = cv2.Canny(gray_img, lower, upper)
    cv2.imshow(f"detect_edges_raw_canny lower={lower}, upper={upper}", edged)
    cv2.waitKey(0)
    return edged


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


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
        if cv2.contourArea(c) < 255:
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
    return cv2.waitKey(0)
