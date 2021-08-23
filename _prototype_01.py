import functions as f
from os import walk


def find_ref_object(path_to_image):
    print(path_to_image)
    img = f.load_image(path_to_image)
    gray = f.bgr_to_gray(img)
    blurred = f.blur_bilateral_filter_min(gray)
    for i in range(35):
        blurred = f.blur_bilateral_filter_min(blurred)
    blurred = f.blur_bilateral_filter_min(blurred)
    binary = f.gray_to_binary(blurred, 77, True)
    edged = f.detect_edges_raw_canny(binary, 25, 100)
    # eroded = f.erode_dilate(edged)
    f.find_contours_and_draw_them(f.gray_to_bgr(gray), edged, path_to_image, 69, 112)


pathToPhotos = "testAlgo1/"
file_names = next(walk(pathToPhotos), (None, None, []))[2]
for file_name in file_names:
    find_ref_object(pathToPhotos + file_name)
# find_ref_object("testAlgo1/0_testAlgo1_0_min_Normal.jpg")
