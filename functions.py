from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

KEY_ESC = 27
KEY_SPACE = 32
KEY_BACKSPACE = 8

REF_OBJ_SIZE_IN_MILLIMETERS = 0.1

PHOTO_WIDTH_PIXELS = 1280
PHOTO_HEIGHT_PIXELS = 1024

ZOOM_IN_MILLIMETER_IN_PIXELS = 706  # 1mm = 706px
ZOOM_IN_WIDTH_MILLIMETERS = PHOTO_WIDTH_PIXELS / ZOOM_IN_MILLIMETER_IN_PIXELS  # = 1,81 mm
ZOOM_IN_HEIGHT_MILLIMETERS = PHOTO_HEIGHT_PIXELS / ZOOM_IN_MILLIMETER_IN_PIXELS  # = 1,45 mm

ZOOM_OUT_MILLIMETER_IN_PIXELS = 100  # 1mm = 100px
ZOOM_OUT_WIDTH_MILLIMETERS = PHOTO_WIDTH_PIXELS / ZOOM_OUT_MILLIMETER_IN_PIXELS  # = 12,8 mm
ZOOM_OUT_HEIGHT_MILLIMETERS = PHOTO_HEIGHT_PIXELS / ZOOM_OUT_MILLIMETER_IN_PIXELS  # = 10,24 mm


def contrast_increase_clahe(img, wait=False):
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
    if wait:
        cv2.imshow("contrast_increase_clahe", final)
        cv2.waitKey(0)
    # _____END_____#
    return final


def sharpen(img, wait=False):
    # Sharpen image
    kernel = np.array([[-1.1, -1.1, -1.1],
                       [-1.1, 9.8, -1.1],
                       [-1.1, -1.1, -1.1]])
    sharpened = cv2.filter2D(img, -1, kernel)  # Applying the sharpening kernel to the input image.
    if wait:
        cv2.imshow("sharpen", sharpened)
        cv2.waitKey(0)
    return sharpened


def gray_to_binary(gray, tresh, wait=False):
    ret, thresholded = cv2.threshold(gray, tresh, 255, cv2.THRESH_BINARY)
    if wait:
        cv2.imshow(f"gray_to_binary tresh: {tresh}", thresholded)
        cv2.waitKey(0)
    return thresholded


def blur_gaussian(gray, wait=False):
    k_size = (7, 7)
    sigma = 0
    # GaussianBlur(src, k_size, sigmaX, dst=None, sigmaY=None, borderType=None)
    blurred = cv2.GaussianBlur(gray, k_size, sigma)
    if wait:
        cv2.imshow(f"blur_gaussian k_size: {k_size}, sigma: {sigma}", gray)
        cv2.waitKey(0)
    return blurred


def blur_bilateral_filter(gray, d, sigma, wait=False):
    d = 4
    sigma = 1
    window_name = f"blur_bilateral_filter d: {d}, sigma: {sigma}"
    blurred = cv2.bilateralFilter(gray, d, sigma, sigma)
    if wait:
        cv2.imshow(window_name, blurred)
        cv2.waitKey(0)
    return blurred


def blur_bilateral_filter_min(gray, wait=False):
    d = 4
    sigma = 11
    window_name = f"blur_bilateral_filter d: {d}, sigma: {sigma}"
    blurred = cv2.bilateralFilter(gray, d, sigma, sigma)
    if wait:
        cv2.imshow(window_name, blurred)
        cv2.waitKey(0)
    return blurred


def load_image(photo_path, wait=False):
    loaded = cv2.imread(photo_path, cv2.IMREAD_COLOR)
    if wait:
        cv2.imshow(f"load_image path:{photo_path}", loaded)
        cv2.waitKey(0)
    return loaded


def bgr_to_gray(bgr_img, wait=False):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    if wait:
        cv2.imshow("bgr_to_gray", gray)
        cv2.waitKey(0)
    return gray


def gray_to_bgr(gray, wait=False):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if wait:
        cv2.imshow("gray_to_bgr", bgr)
        cv2.waitKey(0)
    return bgr


# Find edges using canny edge detector
# noinspection PyTypeChecker
def detect_edges_sigma_v(gray_img, sigma=0.33, v=202, wait=False):
    # compute the median of the single channel pixel intensities
    # v = np.median(grayim)  # 202.0
    # v = np.float64(20)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(gray_img, lower, upper)
    if wait:
        cv2.imshow(f"detect_edges_sigma_v sigma={sigma}, v={v}", edged)
        cv2.waitKey(0)
    return edged


# Find edges using canny edge detector
def detect_edges_auto(gray_img, sigma=0.33, wait=False):
    # compute the median of the single channel pixel intensities
    v = np.median(gray_img)  # 202.0
    # v = np.float64(20)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(gray_img, lower, upper)
    if wait:
        cv2.imshow(f"detect_edges_auto sigma={sigma}, v={v}, lower={lower}, upper={upper}", edged)
        cv2.waitKey(0)
    return edged


# Find edges using canny edge detector
def detect_edges_raw_canny(gray_img, lower, upper, wait=False):
    edged = cv2.Canny(gray_img, lower, upper)
    if wait:
        cv2.imshow(f"detect_edges_raw_canny lower={lower}, upper={upper}", edged)
        cv2.waitKey(0)
    return edged


def erode_dilate(thresholded, wait=False):
    kernel = np.ones((1, 1), np.uint8)
    eroded = cv2.erode(thresholded, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    if wait:
        cv2.imshow("erode_dilate", dilated)
        cv2.waitKey(0)
    return dilated


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def box_area(box):
    box_size = box[1]
    box_x = box_size[0]
    box_y = box_size[1]
    area = box_x * box_y
    return area


def box_size(box):
    return box[1]


def box_center(box):
    return box[0]


def find_contours_and_draw_them(img, edged, window_name, min_size, max_size, show_all=False, wait=False):
    pixels_per_millimeter = None
    # find contours in the edge map
    # cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(edged, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_TC89_L1)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    # loop over the contours individually
    orig = img
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        box = cv2.minAreaRect(c)
        print("box:", box)
        (box_center_x, box_center_y) = box_center(box)
        (box_size_x, box_size_y) = box_size(box)
        area = box_area(box)
        print("area:", area)
        # calculate scale factor form minimum size or from mean of both sizes
        # size_for_scale_calculation = np.min([box_size_x, box_size_y])
        size_for_scale_calculation = np.mean([box_size_x, box_size_y])
        print("size_for_scale_calculation:", size_for_scale_calculation)
        # noinspection PyUnresolvedReferences
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        print("box vertexes:", box)
        is_appropriate_size = min_size < box_size_x < max_size and min_size < box_size_y < max_size
        if pixels_per_millimeter is None:
            if is_appropriate_size:
                # PIXELS_PER_METRIC = dB / REF_OBJ_SIZE_IN_INCH
                pixels_per_millimeter = size_for_scale_calculation / REF_OBJ_SIZE_IN_MILLIMETERS
                # print(f"REF_OBJ_CONTOUR = {c}")
                print(f"REF_OBJ_BOX = {box}")
                print(f"PHOTO_SCALE: 1 mm = {'{:.2f}'.format(pixels_per_millimeter)} px")
        if is_appropriate_size or show_all:
            # sort the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            draw_contour(c, orig)
            draw_box(box, orig)
            draw_corners(box, orig)
            draw_area(area, box_center_x, box_center_y, orig)
            if pixels_per_millimeter is None:
                draw_no_scale(orig, box_center_x, box_center_y)
            else:
                draw_dimensions(box, orig, pixels_per_millimeter)
            if wait:
                cv2.imshow(f"{window_name} min_size = {min_size}", orig)
                cv2.waitKey(0)
            if not show_all:
                break
    cv2.imshow(f"{window_name} min_size = {min_size}, max_size = {max_size}", orig)
    cv2.waitKey(0)
    return pixels_per_millimeter


def draw_box(box, orig):
    # draw box contour
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)


def draw_contour(c, orig):
    # draw contour points
    for [[x, y]] in c:
        cv2.circle(orig, (int(x), int(y)), 1, (0, 127, 255), -1)


def draw_corners(box, orig):
    # draw box corners/vertexes
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 3, (0, 0, 255), -1)


def draw_dimensions(box, orig, pixels_per_millimeter):
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
    draw_object_size(pixels_per_millimeter, dA, dB, orig, tltrX, tltrY, trbrX, trbrY)


def draw_no_scale(orig, box_center_x, box_center_y):
    cv2.putText(orig,
                "NO SCALE",
                (int(box_center_x - 20), int(box_center_y - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2)


def draw_object_size(pixels_per_millimeter, dA, dB, orig, tltrX, tltrY, trbrX, trbrY):
    # compute the size of the object
    dimA = dA / pixels_per_millimeter
    dimB = dB / pixels_per_millimeter
    # draw the object sizes on the image
    # putText(img, text, bottom-left-corner, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)
    cv2.putText(orig,
                "{:.2f} mm".format(dimA),
                (int(trbrX + 10), int(trbrY)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 127),
                1)
    cv2.putText(orig,
                "{:.2f} mm".format(dimB),
                (int(tltrX - 15), int(tltrY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 127),
                1)


def draw_area(area, box_center_x, box_center_y, orig):
    cv2.putText(orig,
                "{:.0f}px2".format(area),
                (int(box_center_x), int(box_center_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1)


def detect_circles(gray, orig):
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="Path to the image")
    # argv = ["", "-istr1/0_str1_0_max_Normal.jpg"]
    # args = vars(ap.parse_args(argv[1:]))
    # # load the image, clone it for output, and then convert it to grayscale
    # image = cv2.imread(args["image"])
    # output = image
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    # cv2.HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1.2, 200, minRadius=40, maxRadius=200)
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
        cv2.imshow(f"detect_circles amount: {circles.size}", orig)
    else:
        cv2.imshow(f"detect_circles NO CIRCLES DETECTED", orig)
    cv2.waitKey(0)
