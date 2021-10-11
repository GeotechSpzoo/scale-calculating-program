import functions as f

ZOOM_IN = "zoom-in"
ZOOM_OUT = "zoom-out"

ZOOM_IN_REF_LINE_LENGTH_IN_MM = 1.4  # millimeters
ZOOM_OUT_REF_LINE_LENGTH_IN_MM = 7 * 1.4  # millimeters

selected_zoom = ZOOM_OUT

MIN_CAL_DOT_SIZE_IN_PX = 90  # pixels

MAX_CAL_DOT_SIZE_IN_PX = 186  # pixels

MIN_DISTANCE_BETWEEN_DOTS_IN_PX = 777  # pixels


def calculate_scale(path_to_photo_folder, main_subject_folder, photo_file_name):
    input_file = path_to_photo_folder + photo_file_name
    print("---------------------------------------")
    print(f"Calculating scale for: {input_file}")
    wait = True
    img = f.load_image(input_file)
    crop = f.crop_dots(img, input_file)
    crop_sample = f.crop_sample(img)
    output_samples_folder = main_subject_folder + "_samples"
    f.save_photo(crop_sample,
                 output_samples_folder,
                 output_samples_folder + "\\" + photo_file_name,
                 override=False)
    gray = f.bgr_to_gray(crop)
    blurred = f.blur_bilateral_filter_min(gray, "")
    for i in range(35):
        blurred = f.blur_bilateral_filter_min(blurred, "")
    blurred = f.blur_bilateral_filter_min(blurred, input_file, wait=wait)
    binary = f.gray_to_binary(blurred, 77, input_file, wait=wait)
    edged = f.detect_edges_raw_canny(binary, 25, 100, input_file)
    contours = f.find_contours(edged, input_file)
    boxes = f.convert_contours_to_min_rect(contours, gray, input_file, wait=wait)
    filtered_boxes = f.filter_boxes_by_size(boxes, MIN_CAL_DOT_SIZE_IN_PX, MAX_CAL_DOT_SIZE_IN_PX, gray,
                                            input_file, wait=wait)
    dot1, dot2 = f.find_ref_dots(filtered_boxes, MIN_DISTANCE_BETWEEN_DOTS_IN_PX, gray, input_file, wait=wait)
    if dot1 is None:
        print("REF OBJECT (calibration_dot1) NOT FOUND!")
        print("Scale calculation aborted!")
        return
    if dot2 is None:
        print("REF OBJECT (calibration_dot2) NOT FOUND!")
        print("Scale calculation aborted!")
        return
    if selected_zoom == ZOOM_IN:
        calculated_scale_one_mm_in_px = f.calculate_and_draw_scale(dot1, dot2, ZOOM_IN_REF_LINE_LENGTH_IN_MM, gray,
                                                                   input_file, wait=True)
    else:
        calculated_scale_one_mm_in_px = f.calculate_and_draw_scale(dot1, dot2, ZOOM_OUT_REF_LINE_LENGTH_IN_MM, gray,
                                                                   input_file, wait=True)
    img_with_a_ruler = f.draw_rulers(img, calculated_scale_one_mm_in_px, input_file, wait=True)
    output_folder = main_subject_folder + "_scale_calculated\\"
    output_file_name = "ruler_" + photo_file_name.replace(".jpg",
                                                          "_{:.0f}dpmm.jpg".format(calculated_scale_one_mm_in_px))
    output_path_to_file = output_folder + output_file_name
    f.save_photo(img_with_a_ruler, output_folder, output_path_to_file, override=True)
    f.exif_copy_all_tags(input_file, output_path_to_file)
    f.exif_update_resolution_tags(output_path_to_file, calculated_scale_one_mm_in_px)
    return calculated_scale_one_mm_in_px


found_jpegs = []

default_path_to_search = "C:\\Users\\pawel.drelich\\Desktop\\Materialy\\AnalizaObrazu\\SamplePhotosLabo\\3144"


def request_path_to_find_photos():
    global found_jpegs
    # pathToPhotos = input("Podej mnie ten ścieżek do zdjęciówek:\n")
    found_jpegs = f.find_all_jpegs(default_path_to_search, ZOOM_OUT)
    return len(found_jpegs)


# iterate through photos
# pathToPhotos = "testTwoDots/"
while request_path_to_find_photos() == 0:
    pass

for (file_folder_path, file_name) in found_jpegs:
    calculate_scale(file_folder_path, default_path_to_search, file_name)

f.close_all_windows()
