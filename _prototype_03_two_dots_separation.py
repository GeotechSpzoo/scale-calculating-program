import numpy as np
from sys import exit

import functions as f

ZOOM_IN = "zoom-in"
ZOOM_OUT = "zoom-out"

ZOOM_IN_REF_LINE_LENGTH_IN_MM = 1.4  # millimeters
ZOOM_OUT_REF_LINE_LENGTH_IN_MM = 7 * 1.4  # millimeters

ZOOM_OUT_MIN_DOT_SIZE_IN_PX = 100  # pixels
ZOOM_OUT_MAX_DOT_SIZE_IN_PX = 186  # pixels

ZOOM_IN_MIN_DOT_SIZE_IN_PX = 90  # pixels
ZOOM_IN_MAX_DOT_SIZE_IN_PX = 186  # pixels

ZOOM_OUT_MIN_DISTANCE_BETWEEN_DOTS_IN_PX = 850  # pixels
ZOOM_OUT_MAX_DISTANCE_BETWEEN_DOTS_IN_PX = 1100  # pixels

ZOOM_IN_MIN_DISTANCE_BETWEEN_DOTS_IN_PX = 1040  # pixels
ZOOM_IN_MAX_DISTANCE_BETWEEN_DOTS_IN_PX = 1090  # pixels

output_folder = ""
output_samples_folder = ""


def calculate_scale(path_to_photo_folder, main_subject_folder, photo_file_name):
    global output_folder, output_samples_folder
    is_zoom_in = ZOOM_IN in file_name
    input_file = path_to_photo_folder + photo_file_name
    print(f"Calculating scale for:\n {input_file}\n")
    wait = False
    img = f.load_image(input_file)
    crop = f.crop_dots(img, input_file)
    crop_sample = f.crop_sample(img, input_file)
    output_samples_folder = main_subject_folder + "_samples"
    output_samples_path_to_file = output_samples_folder + "\\" + photo_file_name
    f.save_photo(crop_sample,
                 output_samples_folder,
                 output_samples_path_to_file,
                 override=False)
    f.exif_copy_all_tags(input_file, output_samples_path_to_file)
    # gray = f.bgr_to_gray(crop)
    gray = f.bgr_to_custom_gray(crop)
    blurred = f.blur_bilateral_filter_min(gray, "")
    blurred = f.contrast_increase_clahe_gray(blurred, wait=wait)
    for i in range(35):
        blurred = f.blur_bilateral_filter_min(blurred, "")
        blurred = f.contrast_increase_clahe_gray(blurred)
    blurred = f.blur_bilateral_filter_min(blurred, input_file, wait=wait)
    # for c in range(0, 20):
    blurred = f.contrast_increase_clahe_gray(blurred, wait=wait)
    dots_found = False
    ref_left_dot, ref_right_dot = None, None
    left_dots, right_dots = [], []
    for tresh in range(0, 100, 5):
        binary = f.gray_to_binary(blurred, input_file, tresh / 100, is_zoom_in, wait=wait)
        edged = f.detect_edges_raw_canny(binary, 25, 100, input_file)
        contours = f.find_contours(edged, input_file)
        boxes = f.convert_contours_to_min_rect(contours, gray, input_file, wait=wait)
        if is_zoom_in:
            filtered_boxes = f.filter_boxes_by_size(boxes, ZOOM_IN_MIN_DOT_SIZE_IN_PX, ZOOM_IN_MAX_DOT_SIZE_IN_PX, gray,
                                                    input_file, wait=wait)
        else:
            filtered_boxes = f.filter_boxes_by_size(boxes, ZOOM_OUT_MIN_DOT_SIZE_IN_PX, ZOOM_OUT_MAX_DOT_SIZE_IN_PX,
                                                    gray,
                                                    input_file, wait=wait)
        left_dot_found = f.find_left_ref_dot(filtered_boxes, gray, is_zoom_in, input_file, wait=wait)
        right_dot_found = f.find_right_ref_dot(filtered_boxes, gray, is_zoom_in, input_file, wait=wait)
        if left_dot_found is not None:
            left_dots.append(left_dot_found)
        elif right_dot_found is not None:
            right_dots.append(right_dot_found)
    if len(left_dots) and len(right_dots) > 0:
        for left_dot in left_dots:
            for right_dot in right_dots:
                distance = f.box_distance(left_dot, right_dot)
                if is_zoom_in:
                    if ZOOM_IN_MIN_DISTANCE_BETWEEN_DOTS_IN_PX < np.abs(
                            distance) < ZOOM_IN_MAX_DISTANCE_BETWEEN_DOTS_IN_PX:
                        ref_left_dot = left_dot
                        ref_right_dot = right_dot
                else:
                    if ZOOM_OUT_MIN_DISTANCE_BETWEEN_DOTS_IN_PX < np.abs(
                            distance) < ZOOM_OUT_MAX_DISTANCE_BETWEEN_DOTS_IN_PX:
                        ref_left_dot = left_dot
                        ref_right_dot = right_dot
    if ref_left_dot and ref_right_dot is not None:
        dots_found = True
    if not dots_found:
        if len(left_dots) == 0:
            print("LEFT CALIBRATION DOT NOT FOUND!")
        if len(right_dots) == 0:
            print("RIGHT CALIBRATION DOT NOT FOUND!")
        print("Scale cannot be calculated.")
        return -1
    if is_zoom_in:
        calculated_scale_one_mm_in_px = f.calculate_scale(ref_left_dot, ref_right_dot, ZOOM_IN_REF_LINE_LENGTH_IN_MM,
                                                          gray,
                                                          input_file, wait=wait)
    else:
        calculated_scale_one_mm_in_px = f.calculate_scale(ref_left_dot, ref_right_dot, ZOOM_OUT_REF_LINE_LENGTH_IN_MM,
                                                          gray,
                                                          input_file, wait=wait)
    img_with_a_ruler = f.draw_rulers(img, ref_left_dot, ref_right_dot, calculated_scale_one_mm_in_px, input_file,
                                     wait=wait,
                                     show_image=wait)
    output_folder = main_subject_folder + "_scale_calculated\\"
    output_file_name = "ruler_" + photo_file_name.replace(".jpg",
                                                          "_{:.0f}dpmm.jpg".format(calculated_scale_one_mm_in_px))
    output_path_to_file = output_folder + output_file_name
    f.save_photo(img_with_a_ruler, output_folder, output_path_to_file, override=True)
    f.exif_copy_all_tags(input_file, output_path_to_file)
    f.exif_update_resolution_tags(output_path_to_file, calculated_scale_one_mm_in_px)
    f.exif_update_resolution_tags(output_samples_path_to_file, calculated_scale_one_mm_in_px)
    f.add_scale_to_file_name(output_samples_path_to_file, calculated_scale_one_mm_in_px)
    return calculated_scale_one_mm_in_px


found_jpegs = []

default_path_to_search = "C:\\Users\\pawel.drelich\\Desktop\\Materialy\\AnalizaObrazu\\SamplePhotosLabo\\3144"
main_folder = ""


def get_input(message):
    result = input(message + "\n")
    if result == "q":
        end_program("Program przerwany przez użytkownika...")
    else:
        return result


def end_program(message):
    finish_message()
    input(message + "\n")
    exit(0)


def request_path_to_find_photos():
    global found_jpegs, main_folder
    main_folder = get_input("Podej mnie ten ścieżek do zdjęciówek:")
    phrase_to_include_in_file_name = get_input(
        "Podej mnie ten frazes, który powinin zawierać się w nazwie plyku, abo walnij ENTERem aby nie flirtować plików:"
    )
    found_jpegs = f.find_all_jpegs(main_folder, phrase_to_include_in_file_name)
    return len(found_jpegs)


# iterate through photos
# pathToPhotos = "testTwoDots/"
# while request_path_to_find_photos() == 0:
#     pass

photos_number_to_proceed = 0
current_photo_index = 0
calculated_photos = []
file_name = ""


def find_photos():
    return request_path_to_find_photos()


def proceed_scale_calculation():
    global file_name, current_photo_index
    for (file_folder_path, file_name) in found_jpegs:
        current_photo_index += 1
        print_line()
        print(f"Photo {current_photo_index} of {photos_number_to_proceed}...")
        calculated_scale = calculate_scale(file_folder_path, main_folder, file_name)
        if calculated_scale != -1:
            calculated_photos.append((file_name, calculated_scale))
    # finish_message()
    f.close_all_windows()
    f.create_report(output_folder, calculated_photos, photos_number_to_proceed)


def finish_message():
    print_line()
    print(f"Scale found in {len(calculated_photos)} of {current_photo_index} processed photos.")
    print(f"Output paths:\n{output_folder}\n{output_samples_folder}")


def print_line():
    print("---------------------------------------")


def start_program():
    global photos_number_to_proceed
    photos_number_to_proceed = find_photos()
    if photos_number_to_proceed > 0:
        get_input("Naciśnij ENTER aby rozpocząć kalkulację skali zdjęć lub wpisz 'q' aby anulować...")
        print("Rozpoczęto analizę zdjęć...")
        proceed_scale_calculation()
        print_line()
        print("Zakończono analizę zdjęć.")
    else:
        print_line()
        print("Nie znaleziono żadnych zdjęć.")
        print(
            "Upewnij się że podana ścieżka jest prawidłowa i spróbuj ponownie lub wpisz 'q' aby wyjść z programu.")
        start_program()


try:
    start_program()
except (Exception, KeyboardInterrupt, OSError) as e:
    print_line()
    print(f"\nERROR\n{e}\nERROR\n")
    if "WinError 2" in str(e):
        print("\tPrawdopodobnie brakuje pliku 'exiftool.exe'. Jest on niezbędny do działania.")
        print("\tŚciągnij go ze strony: https://exiftool.org/ i umieść w katalogu programu.")
    print_line()
    print("Złapano wyjątek. Program został zatrzymany")
finally:
    end_program("Koniec programu...")

# COMPILE COMMAND: 'pyinstaller --onefile --windowed _prototype_03_two_dots_separation.py'
# COMPILE COMMAND: 'pyinstaller --onefile _prototype_03_two_dots_separation.py'
