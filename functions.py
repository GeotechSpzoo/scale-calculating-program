import os

import cv2
import imutils
import numpy as np
from imutils import contours as cont
from imutils import perspective
from scipy.spatial import distance as dist

import exif

KEY_ESC = 27
KEY_SPACE = 32
KEY_BACKSPACE = 8

REF_OBJ_SIZE_IN_MILLIMETERS = 0.1

PHOTO_WIDTH_PIXELS = 1280
PHOTO_HEIGHT_PIXELS = 1024

TWO_DOTS_PX_MIN_DIST = 500  # 1mm = 706px

ZOOM_IN_OUT_FACTOR = 7  # more precise = 7,067

ZOOM_IN_MILLIMETER_IN_PIXELS = 765  # 1mm = 765px
ZOOM_IN_WIDTH_MILLIMETERS = PHOTO_WIDTH_PIXELS / ZOOM_IN_MILLIMETER_IN_PIXELS  # = 1,81 mm
ZOOM_IN_HEIGHT_MILLIMETERS = PHOTO_HEIGHT_PIXELS / ZOOM_IN_MILLIMETER_IN_PIXELS  # = 1,45 mm

ZOOM_OUT_MILLIMETER_IN_PIXELS = 100  # 1mm = 100px
ZOOM_OUT_WIDTH_MILLIMETERS = PHOTO_WIDTH_PIXELS / ZOOM_OUT_MILLIMETER_IN_PIXELS  # = 12,80 mm
ZOOM_OUT_HEIGHT_MILLIMETERS = PHOTO_HEIGHT_PIXELS / ZOOM_OUT_MILLIMETER_IN_PIXELS  # = 10,24 mm


def crop_dots(img, window_name):
    height = img.shape[0]
    width = img.shape[1]
    crop_img = img[0:int(0.2 * height), 0:width]
    # cv2.imshow(f"crop_dots {window_name}", crop_img)
    return crop_img


def crop_dots_with_black_border(img, window_name):
    height = img.shape[0]
    width = img.shape[1]
    crop_height = int(0.2 * height)
    crop_width = width
    blank_image = create_bgr_black_img(crop_height + 6, crop_width + 6)
    crop_img = img[0:int(0.2 * height), 0:width]
    merged = merge(blank_image, crop_img, 3, 3)
    # cv2.imshow(f"crop_dots_with_black_border {window_name}", merged)
    return merged


def create_bgr_black_img(height, width):
    return np.zeros((height, width, 3), np.uint8)


def merge(background, foreground, x, y):
    # get position and crop pasting area if needed
    bg_width = background.shape[0]
    bg_height = background.shape[1]
    fr_width = foreground.shape[0]
    fr_height = foreground.shape[1]
    width = bg_width - x
    height = bg_height - y
    if fr_width < width:
        width = fr_width
    if fr_height < height:
        height = fr_height
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[x:x + width, y:y + height, 2] / 255.0
    alpha_foreground = foreground[:width, :height, 2] / 255.0
    # set adjusted colors
    for color in range(0, 2):
        fr = foreground[:width, :height, color]
        bg = background[x:x + width, y:y + height, color]
        background[x:x + width, y:y + height, color] = fr + bg
    # set adjusted alpha and denormalize back to 0-255
    background[x:x + width, y:y + height, 2] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
    return background


def crop_document(img, window_name):
    height = img.shape[0]
    width = img.shape[1]
    crop_img = img[int(0.25 * height):height, 0:width]
    # cv2.imshow(f"cropped sample {window_name}", crop_img)
    return crop_img


def crop_ai(img, window_name):
    height = img.shape[0]
    width = img.shape[1]
    crop_img = img[int(2 * height / 6):int(5 * height / 6), int(width / 6):int(5 * width / 6)]
    # cv2.imshow(f"cropped sample {window_name}", crop_img)
    return crop_img


def close_all_windows():
    cv2.destroyAllWindows()


def contrast_increase_clahe_gray(img, wait=False):
    final = cv2.equalizeHist(img.copy())
    if wait:
        cv2.imshow("contrast_increase_clahe_gray", final)
        cv2.waitKey(0)
    return final


def contrast_increase_clahe_bgr(img, wait=False):
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
        cv2.imshow("contrast_increase_clahe_bgr", final)
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


def gray_to_binary(gray, path, tresh=0.0, zoom_in=True, wait=False):
    average = gray.mean(axis=0).mean(axis=0)
    if zoom_in:
        ret, thresholded = cv2.threshold(gray, average - tresh * average, 255, cv2.THRESH_BINARY)
    else:
        ret, thresholded = cv2.threshold(gray, average + tresh * average, 255, cv2.THRESH_BINARY)
    if wait:
        cv2.imshow(f"gray_to_binary tresh: {tresh} {path}", thresholded)
        cv2.waitKey(0)
    return thresholded


def gray_to_binary_adaptive(gray, path, block_size=3, tresh=1, zoom_in=True, wait=False):
    average = gray.mean(axis=0).mean(axis=0)
    # block_size = 2 * tresh + 1
    if zoom_in:
        thresholded = cv2.adaptiveThreshold(gray, 255, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY, block_size, tresh)
    else:
        thresholded = cv2.adaptiveThreshold(gray, 255, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY, block_size, tresh)
    if wait:
        cv2.imshow(f"gray_to_binary_adaptive tresh: {tresh} {path}", thresholded)
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


def blur_bilateral_filter_min(gray, path, wait=False):
    d = 4
    sigma = 11
    window_name = f"d: {d}, sigma: {sigma}"
    blurred = cv2.bilateralFilter(gray, d, sigma, sigma)
    if wait:
        cv2.imshow(f"blur_bilateral_filter_min {window_name} {path}", blurred)
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


def bgr_to_custom_gray(bgr_img, wait=False):
    bg_width = bgr_img.shape[0]
    bg_height = bgr_img.shape[1]
    custom_gray = np.zeros((bg_width, bg_height, 1), np.uint8)
    # width = bg_width - x
    # height = bg_height - y
    # if fr_width < width:
    #     width = fr_width
    # if fr_height < height:
    #     height = fr_height
    # normalize alpha channels from 0-255 to 0-1
    # alpha_background = background[x:x + width, y:y + height, 2] / 255.0
    # alpha_foreground = foreground[:width, :height, 2] / 255.0
    # set adjusted colors
    for pxW in range(0, bg_width - 1):
        for pxH in range(0, bg_height - 1):
            b = bgr_img[pxW, pxH, 0]
            g = bgr_img[pxW, pxH, 1]
            r = bgr_img[pxW, pxH, 2]
            # print(f"b: {b} g: {g} r: {r}")
            rgb_sum = int(b) + int(g) + int(r)
            # print(f"rgb_sum: {rgb_sum}")
            custom_gray[pxW, pxH, 0] = rgb_sum / 3
    # set adjusted alpha and denormalize back to 0-255
    # background[x:x + width, y:y + height, 2] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
    if wait:
        cv2.imshow("bgr_to_custom_gray", custom_gray)
        cv2.waitKey(0)
    return custom_gray


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


def detect_edges_raw_canny(gray_img, lower, upper, path, wait=False):
    edged = cv2.Canny(gray_img, lower, upper)
    if wait:
        cv2.imshow(f"detect_edges_raw_canny lower={lower}, upper={upper} {path}", edged)
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


def box_distance(box1, box2):
    return dist.euclidean(box_center(box1), box_center(box2))


def find_biggest_area_box(boxes):
    biggest_area = 0
    biggest_box = None
    for box in boxes:
        area = box_area(box)
        if area > biggest_area:
            biggest_area = area
            biggest_box = box
    return biggest_box


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
    (cnts, _) = cont.sort_contours(cnts)
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
        # box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        # box = np.array(box, dtype="int")
        # print("box vertexes:", box)
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
            # box = perspective.order_points(box)
            draw_contour(c, orig)
            draw_box_with_corners(box, orig)
            draw_area(box, orig)
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


def draw_box_with_corners(box, orig):
    # draw box contour
    box = box_corner_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)
    draw_corners(box, orig)


def box_corner_points(box):
    # noinspection PyUnresolvedReferences
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    return box


def draw_boxes(boxes, orig):
    for box in boxes:
        draw_box_with_corners(box, orig)


def draw_contour(c, orig):
    # draw contour points
    cv2.drawContours(orig, [c], -1, (0, 127, 255), 1)


def draw_contours(contours_list, orig):
    for c in contours_list:
        cv2.drawContours(orig, [c], -1, (0, 127, 255), 1)


def draw_contours_with_label(contours_list, orig):
    # draw contour points
    i = 0
    for c in contours_list:
        # draw the contour and label number on the image
        cv2.drawContours(orig, [c], -1, (0, 127, 255), 1)
        box = cv2.minAreaRect(c)
        (c_x, c_y) = box_center(box)
        cv2.putText(orig, "#{}".format(i + 1), (int(c_x), int(c_y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1)
        i += 1


def draw_corners(box, orig):
    # draw box corners/vertexes
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 3, (0, 0, 255), -1)


def draw_dimensions(box, orig, pixels_per_millimeter):
    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    box = box_corner_points(box)
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
                (0, 255, 0),
                1)
    cv2.putText(orig,
                "{:.2f} mm".format(dimB),
                (int(tltrX - 15), int(tltrY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1)


def draw_area(box, orig):
    (box_center_x, box_center_y) = box_center(box)
    area = box_area(box)
    cv2.putText(orig,
                "{:.0f}px2".format(area),
                (int(box_center_x), int(box_center_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
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


def find_contours(binary_img, window_name, sort="left-to-right", wait=False):
    contours_list = cv2.findContours(binary_img, cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_NONE)
    contours_list = imutils.grab_contours(contours_list)
    if len(contours_list) > 0:
        (contours_list, _) = cont.sort_contours(contours_list, method=sort)
    if wait:
        orig = gray_to_bgr(binary_img)
        draw_contours_with_label(contours_list, orig)
        cv2.imshow(f"find_contours {window_name}", orig)
        cv2.waitKey(0)
    return contours_list


def convert_contours_to_min_rect(contours_list, gray, window_name, wait=False):
    boxes = []
    for c in contours_list:
        boxes.append(cv2.minAreaRect(c))
    if wait:
        orig = gray_to_bgr(gray)
        draw_boxes(boxes, orig)
        cv2.imshow(f"convert_contours_to_min_rect {window_name}", orig)
        cv2.waitKey(0)
    return boxes


def filter_boxes_by_size(boxes, min_size_px, max_size_px, gray, window_name, wait=False):
    filtered_boxes = []
    for box in boxes:
        (box_size_x, box_size_y) = box_size(box)
        if min_size_px < box_size_x < max_size_px and min_size_px < box_size_y < max_size_px:
            filtered_boxes.append(box)
    if wait:
        orig = gray_to_bgr(gray)
        draw_boxes(filtered_boxes, orig)
        cv2.imshow(f"filter_boxes_by_size {window_name}", orig)
        cv2.waitKey(0)
    return filtered_boxes


def image_resolution(orig):
    height, width = orig.shape[:2]
    return height, width


def draw_text_info(orig, text):
    height, width = image_resolution(orig)
    draw_text(orig,
              text,
              int(width / 4),
              int(height / 2),
              (0, 0, 255),
              1.0,
              3)
    # cv2.putText(orig,
    #             text,
    #             (int(width / 4), int(height / 2)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1.0,
    #             (0, 0, 255),
    #             3)


def draw_text(orig, text, x, y, color, size, thickness):
    cv2.putText(orig,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                color,
                thickness)


def find_ref_dots(filtered_boxes, min_px_distance_between_dots, gray, window_name, wait=False):
    calibration_dot1 = None
    calibration_dot2 = None
    for box in filtered_boxes:
        if calibration_dot1 is None:
            calibration_dot1 = box
        else:
            (dot1_x, dot1_y) = box_center(calibration_dot1)
            (x, y) = box_center(box)
            if np.abs(dot1_x - x) > min_px_distance_between_dots:
                calibration_dot2 = box
                break
    orig = gray_to_bgr(gray)
    draw_boxes([calibration_dot1, calibration_dot2], orig)
    if wait:
        if calibration_dot1 is None:
            draw_text_info(orig, "NO CALIBRATION DOTS FOUND")
            cv2.imshow(f"find_ref_dots {window_name}", orig)
            cv2.waitKey(0)
        elif calibration_dot2 is None:
            draw_text_info(orig, "CANT FIND SECOND CALIBRATION DOT")
            cv2.imshow(f"find_ref_dots {window_name}", orig)
            cv2.waitKey(0)
        else:
            cv2.imshow(f"find_ref_dots {window_name}", orig)
            cv2.waitKey(0)
    # print("calibration_dot1:", calibration_dot1)
    # print("calibration_dot2:", calibration_dot2)
    return calibration_dot1, calibration_dot2


def find_left_ref_dot(filtered_boxes, gray, is_zoom_in, window_name, wait=False):
    calibration_dot = None
    height = gray.shape[0]
    width = gray.shape[1]
    for box in filtered_boxes:
        (box_x, box_y) = box_center(box)
        (box_size_x, box_size_y) = box_size(box)
        if box_x > box_size_x / 2 and box_y > height / 4:
            if box_x < width / 5 and box_y < 3 * height / 4:
                calibration_dot = box
                break
    orig = gray_to_bgr(gray)
    draw_box_with_corners(calibration_dot, orig)
    if wait:
        if calibration_dot is None:
            # draw_text_info(orig, "NO LEFT CALIBRATION DOT FOUND")
            # cv2.imshow(f"find_ref_dots {window_name}", orig)
            # cv2.waitKey(0)
            pass
        else:
            cv2.imshow(f"find_left_ref_dot {window_name}", orig)
            cv2.waitKey(0)
    return calibration_dot


def find_right_ref_dot(filtered_boxes, gray, is_zoom_in, window_name, wait=False):
    calibration_dot = None
    height = gray.shape[0]
    width = gray.shape[1]
    for box in filtered_boxes:
        (box_x, box_y) = box_center(box)
        (box_size_x, box_size_y) = box_size(box)
        if box_x > 4 * width / 5 and box_y > height / 4:
            if box_x < width - box_size_x / 2 and box_y < 3 * height / 4:
                calibration_dot = box
                break
    orig = gray_to_bgr(gray)
    draw_box_with_corners(calibration_dot, orig)
    if wait:
        if calibration_dot is None:
            # draw_text_info(orig, "NO LEFT CALIBRATION DOT FOUND")
            # cv2.imshow(f"find_ref_dots {window_name}", orig)
            # cv2.waitKey(0)
            pass
        else:
            cv2.imshow(f"find_right_ref_dot {window_name}", orig)
            cv2.waitKey(0)
    return calibration_dot


def draw_line_with_label(orig, line, label_above, label_under):
    dot1_center = line[0]
    dot2_center = line[1]
    mid_x, mid_y = midpoint(dot1_center, dot2_center)
    cv2.line(orig, (int(dot1_center[0]), int(dot1_center[1])), (int(dot2_center[0]), int(dot2_center[1])),
             (0, 255, 0), 2)
    cv2.putText(orig,
                label_above,
                (int(mid_x - 15), int(mid_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 127, 255),
                2)
    cv2.putText(orig,
                label_under,
                (int(mid_x - 15), int(mid_y + 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 127, 255),
                2)


def calculate_scale(dot1, dot2, ref_dist, gray, window_name, wait=False):
    dot1_center = box_center(dot1)
    dot2_center = box_center(dot2)
    line = (dot1_center, dot2_center)
    line_length = dist.euclidean(dot1_center, dot2_center)
    scale_one_mm_in_px = line_length / ref_dist
    label_above = "calculated_scale 1 mm = {:.2f} px".format(scale_one_mm_in_px)
    print(label_above)
    # print(f"calculate_scaled 100 px = {100 / scale_one_mm_in_px} mm")
    label_under = "line length = {:.2f} px = ".format(line_length) + "{:.2f} mm".format(
        line_length / scale_one_mm_in_px)
    if wait:
        orig = gray_to_bgr(gray)
        draw_line_with_label(orig, line, label_above, label_under)
        cv2.imshow(f"calculate_and_draw_scale {window_name}", orig)
        cv2.waitKey(0)
    return scale_one_mm_in_px


RULER_THICKNESS_VERY_BOLD = 3
RULER_THICKNESS_BOLD = 3
RULER_THICKNESS_NORMAL = 2

RULER_THICKNESS_LIGHT = 1
RULER_LINE_LENGTH_VERY_LONG = 50
RULER_LINE_LENGTH_LONG = 40
RULER_LINE_LENGTH_NORMAL = 30

RULER_LINE_LENGTH_SHORT = 20
RULER_LINE_COLOR = (0, 0, 0)
RULER_LABEL_COLOR = (255, 255, 255)
RULER_LABEL_SIZE = 0.5
RULER_LABEL_THICKNESS = 2

RULER_05_LABEL_LIMIT_PX = 200


def draw_rulers_with_labels(original_img, one_mm_in_px):
    img = original_img.copy()
    i = 0
    counter = 0
    height, width, = image_resolution(img)
    max_length = int(np.maximum(height, width))
    while i < max_length:
        if counter % 100 == 0:
            # horizontal
            cv2.line(img, (int(i), 0), (int(i), RULER_LINE_LENGTH_VERY_LONG), RULER_LABEL_COLOR,
                     RULER_THICKNESS_VERY_BOLD)
            cv2.putText(img, "{:.1f}mm".format(i / one_mm_in_px),
                        (int(i) - RULER_LINE_LENGTH_NORMAL, RULER_LINE_LENGTH_VERY_LONG + RULER_LINE_LENGTH_NORMAL),
                        cv2.FONT_HERSHEY_SIMPLEX, RULER_LABEL_SIZE, RULER_LABEL_COLOR, RULER_LABEL_THICKNESS)
            # vertical
            cv2.line(img, (0, int(i)), (RULER_LINE_LENGTH_VERY_LONG, int(i)), RULER_LABEL_COLOR,
                     RULER_THICKNESS_VERY_BOLD)
            cv2.putText(img, "{:.1f}mm".format(i / one_mm_in_px), (RULER_LINE_LENGTH_VERY_LONG + 10, int(i)),
                        cv2.FONT_HERSHEY_SIMPLEX, RULER_LABEL_SIZE, RULER_LABEL_COLOR, RULER_LABEL_THICKNESS)
        elif counter % 50 == 0:
            # horizontal
            cv2.line(img, (int(i), 0), (int(i), RULER_LINE_LENGTH_VERY_LONG), RULER_LINE_COLOR, RULER_THICKNESS_BOLD)
            if one_mm_in_px > RULER_05_LABEL_LIMIT_PX:
                cv2.putText(img, "{:.2f}mm".format(i / one_mm_in_px),
                            (int(i) - RULER_LINE_LENGTH_NORMAL, RULER_LINE_LENGTH_VERY_LONG + RULER_LINE_LENGTH_NORMAL),
                            cv2.FONT_HERSHEY_SIMPLEX, RULER_LABEL_SIZE, RULER_LABEL_COLOR, RULER_LABEL_THICKNESS)
            # vertical
            cv2.line(img, (0, int(i)), (RULER_LINE_LENGTH_VERY_LONG, int(i)), RULER_LINE_COLOR, RULER_THICKNESS_BOLD)
            cv2.putText(img, "{:.2f}mm".format(i / one_mm_in_px), (RULER_LINE_LENGTH_VERY_LONG + 10, int(i)),
                        cv2.FONT_HERSHEY_SIMPLEX, RULER_LABEL_SIZE, RULER_LABEL_COLOR, RULER_LABEL_THICKNESS)
        elif counter % 10 == 0:
            # horizontal
            cv2.line(img, (int(i), 0), (int(i), RULER_LINE_LENGTH_LONG), RULER_LINE_COLOR, RULER_THICKNESS_NORMAL)
            # vertical
            cv2.line(img, (0, int(i)), (RULER_LINE_LENGTH_LONG, int(i)), RULER_LINE_COLOR, RULER_THICKNESS_NORMAL)
        elif one_mm_in_px > RULER_05_LABEL_LIMIT_PX:
            # horizontal
            cv2.line(img, (int(i), 0), (int(i), RULER_LINE_LENGTH_SHORT), RULER_LINE_COLOR, RULER_THICKNESS_LIGHT)
            # vertical
            cv2.line(img, (0, int(i)), (RULER_LINE_LENGTH_SHORT, int(i)), RULER_LINE_COLOR, RULER_THICKNESS_LIGHT)
        i += one_mm_in_px / 100
        counter += 1
    return img


def draw_calculated_img(original_img, dot1, dot2, scale_one_mm_in_px, window_name, wait=False, show_image=False):
    img = original_img.copy()
    dot1_center = box_center(dot1)
    dot2_center = box_center(dot2)
    line = (dot1_center, dot2_center)
    line_length = dist.euclidean(dot1_center, dot2_center)
    label_above = "calculated_scale 1 mm = {:.2f} px".format(scale_one_mm_in_px)
    label_under = "line length = {:.2f} px = ".format(line_length) + "{:.2f} mm".format(
        line_length / scale_one_mm_in_px)
    img = draw_rulers_with_labels(img, scale_one_mm_in_px)
    draw_line_with_label(img, line, label_above, label_under)
    draw_box_with_corners(dot1, img)
    draw_box_with_corners(dot2, img)
    if show_image:
        cv2.imshow(f"draw_ruler {window_name}", img)
    if wait:
        cv2.waitKey(0)
    return img


def save_photo(img_with_a_ruler, path_to_folder, path_to_file, override=True):
    # path_to_file = folder + file_name
    if not os.path.exists(path_to_folder):
        os.mkdir(path_to_folder)
    if override:
        write_image_to_file(img_with_a_ruler, path_to_file)
    else:
        if os.path.exists(path_to_file):
            # print("Photo file already exist: " + path_to_file)
            # print("Photo not saved!")
            pass
        else:
            write_image_to_file(img_with_a_ruler, path_to_file)
    return path_to_file


def write_image_to_file(img, path_to_file):
    cv2.imwrite(path_to_file, img)
    # print("Photo saved to: " + path_to_file)


def exif_copy_all_tags(source_file, destination_file):
    exif.copy_all_tags(source_file, destination_file)


def exif_get_user_comment(source_file):
    return exif.read_tag_value(exif.user_comment, source_file)


def exif_update_resolution_tags(path_to_file, original_file_path, scale_in_dpmm):
    exif.write_resolution_tags(path_to_file, original_file_path, scale_in_dpmm)


def add_scale_to_file_name(output_samples_path_to_file, calculated_scale_one_mm_in_px):
    new_name = output_samples_path_to_file.replace(".jpg", "_{:.0f}dpmm.jpg".format(calculated_scale_one_mm_in_px))
    if not os.path.exists(new_name):
        os.rename(output_samples_path_to_file, new_name)


def find_all_jpegs(directory, file_name_contains="", show_paths=False):
    print(f"Skanowanie folderów i podfolderów w poszukiwaniu plików '.jpg' zawierających frazę:"
          f" '{file_name_contains}'\n {directory}\n")
    file_counter = 0
    found_jpegs = []
    for root, dirs, files in os.walk(directory):
        if len(files) > 0:
            for file in files:
                file_folder_path = root + os.path.sep
                if str(file).lower().endswith(".jpg") and file_name_contains.lower() in str(file).lower():
                    found_jpegs.append((file_folder_path, file))
                    file_counter += 1
                    if show_paths:
                        print(f"{file_counter}. {file_folder_path + file}")
    print(f"Znaleziono: {file_counter} plików.\n")
    return found_jpegs


def create_report(report_file_path, calculated_photos_with_scale, number_of_proceeded_photos, report_message):
    report_content = f"Skala znaleziona w {len(calculated_photos_with_scale)} z {number_of_proceeded_photos} " \
                     f"przeanalizowanych zdjęć:\n"
    counter = 1
    for (file_name, scale) in calculated_photos_with_scale:
        file_name = file_name.replace(".jpg", "_{:.0f}dpmm.jpg".format(scale))
        report_content = report_content + f"{counter}. {file_name}\n"
        counter += 1
    report_content = report_content + report_message + "\n"
    with open(report_file_path, 'w', encoding="utf-8-sig") as report:
        report.write(report_content)


def draw_documentation_info(img, text):
    height, width = image_resolution(img)
    merged_height = height + 32
    black_img = create_bgr_black_img(merged_height, width)
    merged = merge(black_img, img, 0, 0)
    draw_text(merged, text, 10, int(merged_height - 10), (255, 255, 255), 0.7, 2)
    return merged


def prepare_documentation_info(original_file_path):
    subject_number, research_point_name, depth, humidity, zoom, spectrum = exif.read_user_comment_tags(
        original_file_path)
    subject_number_and_name = exif.read_tag_value(exif.image_description, original_file_path)
    result_text = f"{subject_number_and_name} - {research_point_name} - {depth} - {humidity} - {spectrum}"
    return result_text
