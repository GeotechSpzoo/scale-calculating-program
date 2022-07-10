import os
import pathlib
import traceback

import functions as f

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

default_user_input_folder = "C:\\Users\\pawel.drelich\\Desktop\\Materialy\\AnalizaObrazu\\SamplePhotosLabo\\Alicja"
user_input_folder = ""

ai_output = ""
documentation_output = ""
calculated_output = ""
report_file_path = ""

current_file_name = ""
current_photo_index = 0
number_of_photos_to_proceed = 0

found_jpegs = []
calculated_photos = []

report_message = "Analizę zakończono pomyślnie."
user_abort_message = "Program przerwany po wpisaniu 'q' przez użytkownika."

copy_tags_from_filename = True


def calculate_scale(original_file_folder, main_subject_folder, original_file_name):
    global ai_output, documentation_output
    is_zoom_in = f.ZOOM_IN in current_file_name
    calculated_scale_in_dpmm = -1
    original_file_path = os.path.join(original_file_folder, original_file_name)
    ai_file_folder = main_subject_folder + "_ai"
    documentation_file_folder = main_subject_folder + "_documentation"
    calculated_file_folder = main_subject_folder + "_calculated"
    suffix_for_calculated_file = ".jpg"
    wait = False
    original_comment = f.exif_get_user_comment(original_file_path, from_filename=copy_tags_from_filename,
                                               filename=original_file_name)
    subject_number_with_name = "3144-4 - Aquanet Marlewo"
    # if copy_tags_from_filename:
    #     subject_number_with_name = f.get_subject_full_name(original_comment)
    # else:
    #     subject_number_with_name = f.exif_get_subject_number_with_name(original_file_path)
    print(f"Calculating scale for:\n {original_file_path}\n")
    is_dots_found, original_img, calculated_scale_in_dpmm, suffix_for_calculated_file = image_processing(
        calculated_file_folder, is_zoom_in, original_file_name, original_file_path, calculated_scale_in_dpmm,
        suffix_for_calculated_file, wait, original_comment)
    # documentation output
    documentation_img = f.crop_document(original_img, original_file_path)
    documentation_info = f.prepare_documentation_info(original_comment, subject_number_with_name)
    if is_dots_found:
        documentation_img = f.draw_rulers_with_labels_outside_the_img(documentation_img, calculated_scale_in_dpmm)
        documentation_img = f.draw_documentation_info(documentation_img, documentation_info +
                                                      " : skala obliczona 1mm = {:.0f}px"
                                                      .format(calculated_scale_in_dpmm))
    else:
        if is_zoom_in:
            documentation_img = f.draw_rulers_with_labels_outside_the_img(documentation_img,
                                                                          ZOOM_IN_DEFAULT_SCALE_IN_PIXELS)
            documentation_img = f.draw_documentation_info(documentation_img,
                                                          documentation_info + f" : skala domyslna 1mm = {ZOOM_IN_DEFAULT_SCALE_IN_PIXELS}px +- 3%")
        else:
            documentation_img = f.draw_rulers_with_labels_outside_the_img(documentation_img,
                                                                          ZOOM_OUT_DEFAULT_SCALE_IN_PIXELS)
            documentation_img = f.draw_documentation_info(documentation_img, documentation_info
                                                          + f" : skala domyslna 1mm = {ZOOM_OUT_DEFAULT_SCALE_IN_PIXELS}px +- 13%")
    save_image_with_exif_data(calculated_scale_in_dpmm, documentation_file_folder, documentation_img,
                              original_file_name, original_file_path, suffix_for_calculated_file, original_comment)
    documentation_output = documentation_file_folder
    # ai output
    ai_img = f.crop_ai(original_img, original_file_path)
    save_image_with_exif_data(calculated_scale_in_dpmm, ai_file_folder, ai_img,
                              original_file_name, original_file_path, suffix_for_calculated_file, original_comment)
    ai_output = ai_file_folder
    return calculated_scale_in_dpmm


def image_processing(calculated_file_folder, is_zoom_in, original_file_name, original_file_path,
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
    f.save_photo(img, file_folder, file_path, override=True)
    f.exif_copy_all_tags(original_file_path, file_path)
    if copy_tags_from_filename:
        f.exif_write_user_comment(file_path, original_comment)
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


def request_path_to_find_photos():
    global found_jpegs, user_input_folder
    user_input = sys_get_input("Podej mnie ten ścieżek do zdjęciówek:")
    path = pathlib.Path(user_input)
    try:
        if path.exists() and len(str(path)) > 0 and not path.is_file() and path.is_dir() and str(path) != ".":
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
            print("Podana ścieżka jest nieprawidłowa.")
            return -1
    except Exception:
        print("Podana ścieżka jest nieprawidłowa.")
        return -1


def find_photos():
    i = -1
    while i == -1:
        i = request_path_to_find_photos()
    return i


def proceed_scale_calculation():
    global current_file_name, current_photo_index, calculated_photos
    for (file_folder_path, current_file_name) in found_jpegs:
        current_photo_index += 1
        print_line()
        print(f"Photo {current_photo_index} of {number_of_photos_to_proceed}...")
        calculated_scale = calculate_scale(file_folder_path, user_input_folder, current_file_name)
        if calculated_scale != -1:
            calculated_photos.append((current_file_name, calculated_scale))
    f.close_all_windows()


def finish_message():
    global report_file_path
    print_line()
    print(f"Skala znaleziona w {len(calculated_photos)} z {current_photo_index} przeanalizowanych zdjęć.")
    print(f"Foldery wyjściowe:\n{ai_output}\n{documentation_output}\n{calculated_output}")
    report_file_path = os.path.join(user_input_folder, "report.txt")
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
        proceed_scale_calculation()
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
