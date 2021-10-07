import functions as f

MIN_CAL_DOT_SIZE_ZOOM_IN_PX = 83

MAX_CAL_DOT_SIZE_ZOOM_IN_PX = 108

MIN_PX_DISTANCE_BETWEEN_DOTS_ZOOM_IN = 200

ZOOM_IN_REF_LINE_LENGTH_MM = 1.4


def calculate_scale(folder, name):
    path_to_file = folder + name
    wait = True
    print(path_to_file)
    img = f.load_image(path_to_file)
    gray = f.bgr_to_gray(img)
    blurred = f.blur_bilateral_filter_min(gray, "")
    for i in range(35):
        blurred = f.blur_bilateral_filter_min(blurred, "")
    blurred = f.blur_bilateral_filter_min(blurred, path_to_file, wait=wait)
    binary = f.gray_to_binary(blurred, 77, path_to_file, wait=wait)
    edged = f.detect_edges_raw_canny(binary, 25, 100, path_to_file)
    contours = f.find_contours(edged, path_to_file)
    boxes = f.convert_contours_to_min_rect(contours, gray, path_to_file, wait=wait)
    filtered_boxes = f.filter_boxes_by_size(boxes, MIN_CAL_DOT_SIZE_ZOOM_IN_PX, MAX_CAL_DOT_SIZE_ZOOM_IN_PX, gray,
                                            path_to_file)
    dot1, dot2 = f.find_ref_dots(filtered_boxes, MIN_PX_DISTANCE_BETWEEN_DOTS_ZOOM_IN, gray, path_to_file, wait=wait)
    if dot1 is None:
        print("REF OBJECT (calibration_dot1) NOT FOUND!")
        return
    if dot2 is None:
        print("REF OBJECT (calibration_dot2) NOT FOUND!")
        return
    scale_one_mm_in_px = f.calculate_scale(dot1, dot2, ZOOM_IN_REF_LINE_LENGTH_MM, gray, path_to_file, wait=True)
    img_with_a_ruler = f.draw_rulers(img, scale_one_mm_in_px, path_to_file, wait=True)
    output_folder = "out_" + folder
    output_file_name = "ruler_" + name.replace(".jpg", "_{:.0f}dpmm.jpg".format(scale_one_mm_in_px))
    output_path_to_file = output_folder + output_file_name
    f.save_photo(img_with_a_ruler, output_path_to_file)
    f.update_exif_resolution_tags(output_path_to_file, scale_one_mm_in_px)
    return scale_one_mm_in_px


scale_factor = 0
min_scale_factor = 99999999999999.0
max_scale_factor = 0.0
scale_factor_sum = 0.0
iterations = 0

found_jpegs = []

default_path_to_search = "C:\\Users\\pawel.drelich\\Desktop\\Materialy\\AnalizaObrazu\\SamplePhotosLabo\\3144"


def request_path_to_find_photos():
    global found_jpegs
    # pathToPhotos = input("Podej mnie ten ścieżek do zdjęciówek:\n")
    found_jpegs = f.find_all_jpegs(default_path_to_search, True)
    return len(found_jpegs)


# iterate through photos
# pathToPhotos = "testTwoDots/"
while request_path_to_find_photos() == 0:
    pass

for (path, file_name) in found_jpegs:
    print(path)
    scale_factor = calculate_scale(path, file_name)
#     # find minimum and max scale_factor
#     if scale_factor is not None:
#         scale_factor_sum += scale_factor
#         iterations += 1
#         if scale_factor > max_scale_factor:
#             max_scale_factor = scale_factor
#         if scale_factor < min_scale_factor:
#             min_scale_factor = scale_factor

# print(f"min_scale_factor: 1 mm = {min_scale_factor} px")
# print(f"max_scale_factor: 1 mm = {max_scale_factor} px")
# average_scale_factor = scale_factor_sum / iterations
# print(f"average_scale_factor: 1 mm = {average_scale_factor} px")
# min_max_deviation = max_scale_factor / min_scale_factor - 1
# print(f"min_max_deviation: {min_max_deviation * 100} %")
# mean_deviation = np.max(
#     [np.abs(average_scale_factor - min_scale_factor), np.abs(average_scale_factor - max_scale_factor)])
# print(f"mean_deviation: {mean_deviation / average_scale_factor * 100} %")
# calculate_scale("testTwoDots/2_testAlgo1_0_min_IR.jpg")
