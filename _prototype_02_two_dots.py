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
    is_zoom_in = ZOOM_IN in current_file_name
    input_file = path_to_photo_folder + photo_file_name
    print(f"Calculating scale for: {input_file}")
    wait = False
    img = f.load_image(input_file)
    crop = f.crop_dots(img, input_file)
    crop_sample = f.crop_document(img, input_file)
    output_samples_folder = main_subject_folder + "_samples"
    f.save_photo(crop_sample,
                 output_samples_folder,
                 output_samples_folder + "\\" + photo_file_name,
                 override=False)
    # gray = f.bgr_to_gray(crop)
    gray = f.bgr_to_custom_gray(crop)
    blurred = f.blur_bilateral_filter_min(gray, "")
    blurred = f.contrast_increase_clahe_gray(blurred, wait=True)
    for i in range(35):
        blurred = f.blur_bilateral_filter_min(blurred, "")
        blurred = f.contrast_increase_clahe_gray(blurred)
    blurred = f.blur_bilateral_filter_min(blurred, input_file, wait=True)
    # for c in range(0, 20):
    blurred = f.contrast_increase_clahe_gray(blurred, wait=True)
    dots_found = False
    dot1, dot2 = None, None
    for tresh in range(0, 100, 5):
        binary = f.gray_to_binary(blurred, input_file, tresh / 100, is_zoom_in, wait=True)
        edged = f.detect_edges_raw_canny(binary, 25, 100, input_file)
        contours = f.find_contours(edged, input_file)
        boxes = f.convert_contours_to_min_rect(contours, gray, input_file, wait=True)
        filtered_boxes = f.filter_boxes_by_size(boxes, MIN_CAL_DOT_SIZE_IN_PX, MAX_CAL_DOT_SIZE_IN_PX, gray,
                                                input_file, wait=wait)
        dot1, dot2 = f.find_ref_dots(filtered_boxes, MIN_DISTANCE_BETWEEN_DOTS_IN_PX, gray, input_file, wait=wait)
        if dot1 is None:
            continue
        elif dot2 is None:
            continue
        else:
            dots_found = True
            break
    if not dots_found:
        if dot1 is None:
            abort_message("REF OBJECT (calibration_dot1) NOT FOUND!")
        elif dot2 is None:
            abort_message("REF OBJECT (calibration_dot2) NOT FOUND!")
        return -1
    if is_zoom_in:
        calculated_scale_in_dpmm = f.calculate_scale(dot1, dot2, ZOOM_IN_REF_LINE_LENGTH_IN_MM, gray,
                                                     input_file, wait=wait)
    else:
        calculated_scale_in_dpmm = f.calculate_scale(dot1, dot2, ZOOM_OUT_REF_LINE_LENGTH_IN_MM, gray,
                                                     input_file, wait=wait)
    img_with_a_ruler = f.draw_calculated_img(img, dot1, dot2, calculated_scale_in_dpmm, input_file, wait=wait,
                                             show_image=wait)
    output_folder = main_subject_folder + "_scale_calculated\\"
    output_file_name = "ruler_" + photo_file_name.replace(".jpg",
                                                          "_{:.0f}dpmm.jpg".format(calculated_scale_in_dpmm))
    output_path_to_file = output_folder + output_file_name
    f.save_photo(img_with_a_ruler, output_folder, output_path_to_file, override=True)
    f.exif_copy_all_tags(input_file, output_path_to_file)
    f.exif_update_resolution_tags(output_path_to_file, input_file, calculated_scale_in_dpmm)
    return calculated_scale_in_dpmm


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

number_of_photos_to_proceed = 0
current_photo_index = 1
calculated_photos = []


def find_photos():
    global number_of_photos_to_proceed
    photos_number_to_proceed = request_path_to_find_photos()


find_photos()

for (file_folder_path, current_file_name) in found_jpegs:
    print("---------------------------------------")
    print(f"Photo {current_photo_index} of {number_of_photos_to_proceed}...")
    calculated_scale = calculate_scale(file_folder_path, default_path_to_search, current_file_name)
    current_photo_index += 1
    if calculated_scale != -1:
        calculated_photos.append((current_file_name, calculated_scale))

print("---------------------------------------")
print(f"{len(calculated_photos)} of {number_of_photos_to_proceed} photos calculated.")
print(f"Output paths:\n{output_folder}\n{output_samples_folder}")

f.close_all_windows()
