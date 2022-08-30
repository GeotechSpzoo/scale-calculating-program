import os
from pathlib import Path
import traceback

import functions as f

SCALE_NOT_CALCULATED = -1

CALCULATED_FOLDER_SUFFIX = "_calculated"

DOCUMENTATION_FOLDER_SUFFIX = "_documentation"

AI_FOLDER_SUFFIX = "_ai"

REPORT_FILE_NAME = "report.txt"

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

ZOOM_IN_DEFAULT_SCALE_IN_PIXELS = 765  # 1mm = 765px
ZOOM_OUT_DEFAULT_SCALE_IN_PIXELS = 100  # 1mm = 100px

# D:\Praca\Geotech\Zdalna_listopad-luty2022\Metadane\339_T
default_user_input_folder = "D:\\Praca\\Geotech\\Zdalna_listopad-luty2022\\Metadane\\339_T"
user_input_folder = ""

ai_output = ""
documentation_output = ""
calculated_output = ""
report_file_path = ""

current_photo_index = 0
number_of_photos_to_proceed = 0

found_jpegs = []
calculated_photos = []

report_message = "Analizę zakończono pomyślnie."
user_abort_message = "Program przerwany po wpisaniu 'q' przez użytkownika."

WAIT = False
is_exif_comment_tags_empty = False
is_exif_subject_number_with_name_empty = False

# User selection
is_override_original_photo_metadata_enabled = True
is_override_existing_output_files_enabled = True
is_override_subject_number_with_name_enabled = False
is_retrieve_metadata_if_possible_enabled = True
is_scale_calculation_enabled = True
is_ai_output_enabled = True
is_documentation_output_enabled = True
# C:\Users\pawel.drelich\Desktop\AUTOMATYCZNE_ZESTAWIANIE_ZDJEC\dlaKonrada\zdjecia1
# subject_number_with_name_to_override = "3144 - Aquanet Marlewo"
# C:\Users\pawel.drelich\Desktop\AUTOMATYCZNE_ZESTAWIANIE_ZDJEC\dlaKonrada\zdjecia2
# subject_number_with_name_to_override = "3203 - DK16 Giby-Ogrodniki"
# C:\Users\pawel.drelich\Desktop\AUTOMATYCZNE_ZESTAWIANIE_ZDJEC\dlaKonrada\zdjecia3
subject_number_with_name_to_override = "3202 DK 16 Gleboki Brod - Giby"


# C:\Users\pawel.drelich\Desktop\AUTOMATYCZNE_ZESTAWIANIE_ZDJEC\dlaKonrada\zdjecia2\109_NAW\5.0m\zoom-in\NAT\WN\
# Warning: FileName encoding not specified


def calculate_scale(found_jpeg_path: Path, main_subject_folder):
    global ai_output, documentation_output, is_exif_comment_tags_empty, is_exif_subject_number_with_name_empty
    print(f"Processing:\n {found_jpeg_path}\n")
    is_zoom_in = f.ZOOM_IN in str(found_jpeg_path)
    calculated_scale_in_dpmm = SCALE_NOT_CALCULATED
    ai_folder_path = Path(main_subject_folder + AI_FOLDER_SUFFIX)
    documentation_folder_path = Path(main_subject_folder + DOCUMENTATION_FOLDER_SUFFIX)
    calculated_folder_path = Path(main_subject_folder + CALCULATED_FOLDER_SUFFIX)
    suffix_for_calculated_file = ".jpg"
    original_comment, subject_number_with_name = get_metadata(found_jpeg_path)
    if is_retrieve_metadata_if_possible_enabled:
        if is_exif_comment_tags_empty or is_exif_subject_number_with_name_empty:
            print(f"Retrieving metadata...")
            f.exif_rewrite_all_exif_metadata(found_jpeg_path, original_comment, subject_number_with_name)
            print(f"Metadata retrieved: subject={subject_number_with_name}, tags={original_comment}")
        else:
            pass  # print(f"Metadata exist: subject={subject_number_with_name}, tags={original_comment}")
    if is_scale_calculation_enabled:
        is_dots_found, original_img, calculated_scale_in_dpmm, suffix_for_calculated_file = proceed_scale_calculation(
            calculated_folder_path, is_zoom_in, found_jpeg_path.name, found_jpeg_path, calculated_scale_in_dpmm,
            suffix_for_calculated_file, WAIT, original_comment)
        if is_documentation_output_enabled:
            proceed_documentation_output(calculated_scale_in_dpmm, documentation_folder_path, found_jpeg_path,
                                         is_zoom_in, original_comment, original_img, subject_number_with_name,
                                         suffix_for_calculated_file, is_dots_found)
        if is_ai_output_enabled:
            proceed_ai_ouput(ai_folder_path, calculated_scale_in_dpmm, found_jpeg_path, original_comment, original_img,
                             suffix_for_calculated_file)
    is_exif_comment_tags_empty = False
    is_exif_subject_number_with_name_empty = False
    return calculated_scale_in_dpmm


def proceed_documentation_output(calculated_scale_in_dpmm, documentation_folder_path, found_jpeg_path,
                                 is_zoom_in, original_comment, original_img, subject_number_with_name,
                                 suffix_for_calculated_file, is_dots_found=False):
    global documentation_output
    # documentation output
    documentation_img = f.crop_document(original_img, found_jpeg_path)
    documentation_info = f.prepare_documentation_info(original_comment, subject_number_with_name)
    if is_dots_found:
        documentation_img = f.draw_rulers_with_labels_outside_the_img(documentation_img, calculated_scale_in_dpmm)
        documentation_img = f.draw_documentation_info(documentation_img,
                                                      prepare_documentation_legend_info_text_calculated_scale(
                                                          calculated_scale_in_dpmm, documentation_info))
    else:
        if is_zoom_in:
            documentation_img = f.draw_rulers_with_labels_outside_the_img(documentation_img,
                                                                          ZOOM_IN_DEFAULT_SCALE_IN_PIXELS)
            documentation_img = f.draw_documentation_info(documentation_img,
                                                          prepare_documentation_legend_info_text_zoom_in(
                                                              documentation_info))
        else:
            documentation_img = f.draw_rulers_with_labels_outside_the_img(documentation_img,
                                                                          ZOOM_OUT_DEFAULT_SCALE_IN_PIXELS)
            documentation_img = f.draw_documentation_info(documentation_img,
                                                          prepare_documentation_legend_info_text_zoom_out(
                                                              documentation_info))
    save_image_with_exif_data(calculated_scale_in_dpmm, documentation_folder_path, documentation_img,
                              found_jpeg_path.name, found_jpeg_path, suffix_for_calculated_file, original_comment)
    documentation_output = documentation_folder_path


def proceed_ai_ouput(ai_folder_path, calculated_scale_in_dpmm, found_jpeg_path, original_comment, original_img,
                     suffix_for_calculated_file):
    global ai_output
    # ai output
    ai_img = f.crop_ai(original_img, found_jpeg_path)
    save_image_with_exif_data(calculated_scale_in_dpmm, ai_folder_path, ai_img,
                              found_jpeg_path.name, found_jpeg_path, suffix_for_calculated_file, original_comment)
    ai_output = ai_folder_path


def get_metadata(found_jpeg_path):
    global is_exif_comment_tags_empty, is_exif_subject_number_with_name_empty
    original_comment = f.exif_get_user_comment(found_jpeg_path)
    print(f"original_comment={original_comment}")
    subject_number_with_name = f.exif_get_subject_number_with_name(found_jpeg_path)
    print(f"subject_number_with_name={subject_number_with_name}")
    # original_comment = ""
    # subject_number_with_name = ""
    if is_retrieve_metadata_if_possible_enabled:
        if original_comment == "" or not original_comment:
            # f.prepare_comment_tags_from_filename(source_file.name)
            is_exif_comment_tags_empty = True
            original_comment = f.prepare_comment_tags_from_path(found_jpeg_path)
        if subject_number_with_name == "" or not subject_number_with_name:
            is_exif_subject_number_with_name_empty = True
            if is_override_subject_number_with_name_enabled:
                subject_number_with_name = subject_number_with_name_to_override
            else:
                subject_number_with_name = f.retrieve_subject_number(found_jpeg_path)
    return original_comment, subject_number_with_name


def prepare_documentation_legend_info_text_calculated_scale(calculated_scale_in_dpmm, documentation_info):
    return documentation_info + " : skala obliczona 1mm = {:.0f}px".format(calculated_scale_in_dpmm)


def prepare_documentation_legend_info_text_zoom_in(documentation_info):
    return documentation_info + f" : skala domyslna 1mm = {ZOOM_IN_DEFAULT_SCALE_IN_PIXELS}px +- 3%"


def prepare_documentation_legend_info_text_zoom_out(documentation_info):
    return documentation_info + f" : skala domyslna 1mm = {ZOOM_OUT_DEFAULT_SCALE_IN_PIXELS}px +- 13%"


def proceed_scale_calculation(calculated_file_folder, is_zoom_in, original_file_name, original_file_path,
                              scale_calculated_one_mm_in_px, suffix_for_calculated_file, wait, original_comment):
    original_img = f.load_image(original_file_path)
    original_img_dots = f.crop_dots(original_img, original_file_path)
    img_gray = f.bgr_to_custom_gray(original_img_dots)
    img_blurred = f.blur_bilateral_filter_min(img_gray, "")
    img_blurred = f.contrast_increase_clahe_gray(img_blurred, wait=wait)
    for i in range(35):
        img_blurred = f.blur_bilateral_filter_min(img_blurred, "")
        img_blurred = f.contrast_increase_clahe_gray(img_blurred)
    img_blurred = f.blur_bilateral_filter_min(img_blurred, original_file_path, wait=wait)
    img_blurred = f.contrast_increase_clahe_gray(img_blurred, wait=wait)
    is_dots_found = False
    left_dots, right_dots = [], []
    ref_left_dot, ref_right_dot = find_ref_calibration_dots(img_blurred, img_gray, original_file_path, is_zoom_in,
                                                            left_dots, right_dots, wait)
    if ref_left_dot and ref_right_dot is not None:
        is_dots_found = True
    if not is_dots_found:
        if len(left_dots) == 0:
            print("LEFT CALIBRATION DOT NOT FOUND!")
        if len(right_dots) == 0:
            print("RIGHT CALIBRATION DOT NOT FOUND!")
        print("Scale cannot be calculated.")
    else:
        if is_zoom_in:
            scale_calculated_one_mm_in_px = f.calculate_scale(ref_left_dot, ref_right_dot,
                                                              ZOOM_IN_REF_LINE_LENGTH_IN_MM, img_gray,
                                                              original_file_path, wait=wait)
        else:
            scale_calculated_one_mm_in_px = f.calculate_scale(ref_left_dot, ref_right_dot,
                                                              ZOOM_OUT_REF_LINE_LENGTH_IN_MM, img_gray,
                                                              original_file_path, wait=wait)
        suffix_for_calculated_file = "_{:.0f}dpmm.jpg".format(scale_calculated_one_mm_in_px)
        # calculated output
        calculated_img = f.draw_calculated_img(original_img, ref_left_dot, ref_right_dot, scale_calculated_one_mm_in_px,
                                               original_file_path, wait=wait, show_image=wait)
        save_image_with_exif_data(scale_calculated_one_mm_in_px, calculated_file_folder, calculated_img,
                                  original_file_name, original_file_path, suffix_for_calculated_file, original_comment)
        global calculated_output
        calculated_output = calculated_file_folder
    return is_dots_found, original_img, scale_calculated_one_mm_in_px, suffix_for_calculated_file


def save_image_with_exif_data(scale_calculated_one_mm_in_px,
                              file_folder,
                              img,
                              original_file_name,
                              original_file_path,
                              suffix_for_calculated_file,
                              original_comment):
    file_path = os.path.join(file_folder, original_file_name.replace(".jpg", suffix_for_calculated_file))
    f.save_photo(img, file_folder, file_path, override=is_override_existing_output_files_enabled)
    f.exif_copy_all_tags(original_file_path, file_path)
    f.exif_update_resolution_tags(file_path, scale_calculated_one_mm_in_px, original_comment)


def find_ref_calibration_dots(blurred, gray, input_file, is_zoom_in, left_dots, right_dots,
                              wait):
    result_ref_left_dot, result_ref_right_dot = None, None
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
                    if ZOOM_IN_MIN_DISTANCE_BETWEEN_DOTS_IN_PX < f.math_abs(
                            distance) < ZOOM_IN_MAX_DISTANCE_BETWEEN_DOTS_IN_PX:
                        result_ref_left_dot = left_dot
                        result_ref_right_dot = right_dot
                else:
                    if ZOOM_OUT_MIN_DISTANCE_BETWEEN_DOTS_IN_PX < f.math_abs(
                            distance) < ZOOM_OUT_MAX_DISTANCE_BETWEEN_DOTS_IN_PX:
                        result_ref_left_dot = left_dot
                        result_ref_right_dot = right_dot
    return result_ref_left_dot, result_ref_right_dot


def sys_get_input(message):
    global user_abort_message
    result = input(message + "\n").strip()
    if result == "q":
        raise Exception(user_abort_message)
    else:
        return result


def sys_end_program(message):
    finish_message()
    input(message + "\n")
    f.sys_exit()


def verify_path_for_exiftool(path):
    sample = f.exif_get_user_comment(path)
    return "Warning: FileName encoding not specified" not in sample


def request_path_to_find_photos():
    global found_jpegs, user_input_folder
    user_input = sys_get_input("Podej mnie ten ścieżek do zdjęciówek:")
    path = Path(user_input)
    try:
        if path.exists() and len(str(path)) > 0 and not path.is_file() and path.is_dir() and str(path) != ".":
            is_path_verified = verify_path_for_exiftool(path)
            if is_path_verified:
                user_input_folder = str(os.path.abspath(path))
                phrase_to_include_in_file_name = sys_get_input(
                    "Podej mnie ten frazes, który powinin zawierać się w nazwie plyku,"
                    " abo walnij ENTERem aby nie flirtować plików:")
                found_jpegs = f.find_all_jpegs(user_input_folder, phrase_to_include_in_file_name)
                if found_jpegs is None:
                    return 0
                else:
                    return len(found_jpegs)
            else:
                print("Podana ścieżka posiada niedozwolone znaki, np. polskie litery. Usuń je!")
        else:
            print("Podana ścieżka jest nieprawidłowa. Sprawdź czy nie ma w niej np. spacji lub polskich liter.")
    except Exception:
        print("Podana ścieżka jest nieprawidłowa.")
    return -1


def find_photos():
    i = -1
    while i == -1:
        i = request_path_to_find_photos()
    return i


def start_scale_calculation():
    global current_photo_index, calculated_photos
    for found_jpeg in found_jpegs:
        current_photo_index += 1
        print_line()
        print(f"Photo {current_photo_index} of {number_of_photos_to_proceed}...")
        calculated_scale = calculate_scale(found_jpeg, user_input_folder)
        if calculated_scale != -1:
            calculated_photos.append((found_jpeg, calculated_scale))
    f.close_all_windows()


def finish_message():
    global report_file_path
    print_line()
    print(f"Skala znaleziona w {len(calculated_photos)} z {current_photo_index} przeanalizowanych zdjęć.")
    print(f"Foldery wyjściowe:\n{ai_output}\n{documentation_output}\n{calculated_output}")
    report_file_path = os.path.join(user_input_folder, REPORT_FILE_NAME)
    report_file_path = os.path.abspath(report_file_path)
    f.create_report(report_file_path, calculated_photos, current_photo_index, report_message)
    print(f"Raport:\n{report_file_path}")


def print_line():
    print("---------------------------------------")


def sys_start_program():
    global number_of_photos_to_proceed
    number_of_photos_to_proceed = find_photos()
    if number_of_photos_to_proceed > 0:
        sys_get_input("Naciśnij ENTER aby rozpocząć kalkulację skali zdjęć lub wpisz 'q' aby anulować...")
        print("Rozpoczęto analizę zdjęć...")
        start_scale_calculation()
        print_line()
        print("Zakończono analizę zdjęć.")
    else:
        print_line()
        print("Nie znaleziono żadnych zdjęć.")
        print(
            "Upewnij się że podana ścieżka jest prawidłowa i spróbuj ponownie lub wpisz 'q' aby wyjść z programu.")
        sys_start_program()


# testing code
# sys_start_program()
# sys_end_program("test end")


f.prepare_working_dir()
try:
    sys_start_program()
except (Exception, KeyboardInterrupt, OSError) as e:
    if str(e) == user_abort_message:
        report_message = str(e)
        print_line()
        print(e)
    else:
        report_message = f"Przerwano działanie programu z powodu: {e}"
        print_line()
        if len(str(e)) == 0:
            report_message = "Wciśnięto CTRL + C lub przerwano działanie programu z nieznanego powodu."
            e = report_message
        elif "WinError 2" in str(e) and "'" not in str(e):
            report_message = f"Prawdopodobnie brakuje pliku 'exiftool.exe'. Przerwano działanie programu: {e}"
            print("\tPrawdopodobnie brakuje pliku 'exiftool.exe'. Jest on niezbędny do działania programu.")
            print("\tŚciągnij go ze strony: https://exiftool.org/ i umieść w katalogu programu.")
        print(f"\nERROR:\n{e}\n")
        print_line()
        traceback = traceback.format_exc()
        report_message = f"{report_message}\n{traceback}"
        print(traceback)
        print_line()
        print("Złapano wyjątek. Program został zatrzymany.")
finally:
    sys_end_program("Koniec programu...")

# COMPILE COMMAND: 'pyinstaller --onefile --windowed _prototype_04_two_dots_separation.py'
# COMPILE COMMAND: 'pyinstaller --onefile _prototype_04_two_dots_separation.py'
# python -m pip install [packagename]
# py -m pip install [packagename]
