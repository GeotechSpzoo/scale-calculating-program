import functions as f
from os import walk


def find_ref_object(path_to_image):
    print(path_to_image)
    img = f.load_image(path_to_image)
    gray = f.bgr_to_gray(img)
    blurred = f.blur_bilateral_filter_min(gray)
    for i in range(35):
        blurred = f.blur_bilateral_filter_min(blurred)
    binary = f.gray_to_binary(blurred, 77)
    edged = f.detect_edges_raw_canny(binary, 25, 100)
    # eroded = f.erode_dilate(edged)
    return f.find_contours_and_draw_them(f.gray_to_bgr(gray), edged, path_to_image, 83, 108)


scale_factor = 0
min_scale_factor = 99999999999999.0
max_scale_factor = 0.0

# iterate through photos
pathToPhotos = "testAlgo1/"
file_names = next(walk(pathToPhotos), (None, None, []))[2]
for file_name in file_names:
    scale_factor = find_ref_object(pathToPhotos + file_name)
    # find minimum and max scale_factor
    if scale_factor is not None:
        if scale_factor > max_scale_factor:
            max_scale_factor = scale_factor
        if scale_factor < min_scale_factor:
            min_scale_factor = scale_factor

print(f"min_scale_factor: {min_scale_factor}")
print(f"max_scale_factor: {max_scale_factor}")
scale_deviation = max_scale_factor / min_scale_factor
print(f"scale deviation: {scale_deviation - 1}")
# find_ref_object("testAlgo1/0_testAlgo1_0_min_Normal.jpg")
