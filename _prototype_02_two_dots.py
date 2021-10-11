import functions as f

ZOOM_IN = "zoom-in"
ZOOM_OUT = "zoom-out"

ZOOM_IN_REF_LINE_LENGTH_IN_MM = 1.4  # millimeters
ZOOM_OUT_REF_LINE_LENGTH_IN_MM = 7 * 1.4  # millimeters

MIN_CAL_DOT_SIZE_IN_PX = 90  # pixels

MAX_CAL_DOT_SIZE_IN_PX = 186  # pixels

MIN_DISTANCE_BETWEEN_DOTS_IN_PX = 777  # pixels

output_folder = ""
output_samples_folder = ""


def calculate_scale(path_to_photo_folder, main_subject_folder, photo_file_name):
    global output_folder, output_samples_folder
    is_zoom_in = ZOOM_IN in file_name
    input_file = path_to_photo_folder + photo_file_name
    print(f"Calculating scale for: {input_file}")
    wait = False
    img = f.load_image(input_file)
    crop = f.crop_dots(img, input_file)
    crop_sample = f.crop_sample(img, input_file)
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
    binary = f.gray_to_binary(blurred, 77, input_file, is_zoom_in, wait=wait)
    edged = f.detect_edges_raw_canny(binary, 25, 100, input_file)
    contours = f.find_contours(edged, input_file)
    boxes = f.convert_contours_to_min_rect(contours, gray, input_file, wait=wait)
    filtered_boxes = f.filter_boxes_by_size(boxes, MIN_CAL_DOT_SIZE_IN_PX, MAX_CAL_DOT_SIZE_IN_PX, gray,
                                            input_file, wait=wait)
    dot1, dot2 = f.find_ref_dots(filtered_boxes, MIN_DISTANCE_BETWEEN_DOTS_IN_PX, gray, input_file, wait=wait)
    if dot1 is None:
        abort_message("REF OBJECT (calibration_dot1) NOT FOUND!")
        return -1
    if dot2 is None:
        abort_message("REF OBJECT (calibration_dot2) NOT FOUND!")
        return -1
    if is_zoom_in:
        calculated_scale_one_mm_in_px = f.calculate_scale(dot1, dot2, ZOOM_IN_REF_LINE_LENGTH_IN_MM, gray,
                                                          input_file, wait=wait)
    else:
        calculated_scale_one_mm_in_px = f.calculate_scale(dot1, dot2, ZOOM_OUT_REF_LINE_LENGTH_IN_MM, gray,
                                                          input_file, wait=wait)
    img_with_a_ruler = f.draw_rulers(img, dot1, dot2, calculated_scale_one_mm_in_px, input_file, wait=wait)
    output_folder = main_subject_folder + "_scale_calculated\\"
    output_file_name = "ruler_" + photo_file_name.replace(".jpg",
                                                          "_{:.0f}dpmm.jpg".format(calculated_scale_one_mm_in_px))
    output_path_to_file = output_folder + output_file_name
    f.save_photo(img_with_a_ruler, output_folder, output_path_to_file, override=True)
    f.exif_copy_all_tags(input_file, output_path_to_file)
    f.exif_update_resolution_tags(output_path_to_file, calculated_scale_one_mm_in_px)
    return calculated_scale_one_mm_in_px


def abort_message(message):
    print(message)
    print("Scale cannot be calculated.")


found_jpegs = []

default_path_to_search = "C:\\Users\\pawel.drelich\\Desktop\\Materialy\\AnalizaObrazu\\SamplePhotosLabo\\3144"


def request_path_to_find_photos():
    global found_jpegs
    # pathToPhotos = input("Podej mnie ten ścieżek do zdjęciówek:\n")
    found_jpegs = f.find_all_jpegs(default_path_to_search)
    return len(found_jpegs)


# iterate through photos
# pathToPhotos = "testTwoDots/"
# while request_path_to_find_photos() == 0:
#     pass

photos_number_to_proceed = 0
current_photo_index = 1
calculated_photos = []


def find_photos():
    global photos_number_to_proceed
    photos_number_to_proceed = request_path_to_find_photos()


find_photos()

for (file_folder_path, file_name) in found_jpegs:
    print("---------------------------------------")
    print(f"Photo {current_photo_index} of {photos_number_to_proceed}...")
    calculated_scale = calculate_scale(file_folder_path, default_path_to_search, file_name)
    current_photo_index += 1
    if calculated_scale != -1:
        calculated_photos.append((file_name, calculated_scale))

print("---------------------------------------")
print(f"{len(calculated_photos)} of {photos_number_to_proceed} photos calculated.")
print(f"Output paths:\n{output_folder}\n{output_samples_folder}")

f.close_all_windows()
