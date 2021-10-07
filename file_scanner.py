import os


def find_all_jpegs(directory, show_file_paths=False):
    files_paths = next(os.walk(directory), (None, None, []))[2]
    jpegs = []
    jpeg_counter = 0
    for file in files_paths:
        if file.endswith(".jpg"):
            jpegs.append(file)
            jpeg_counter += 1
            if show_file_paths:
                print(f"{jpeg_counter}. {file}")
    print(f"Znaleziono: {jpeg_counter} plików .jpg")
    return jpegs


def find_all_jpegs_2(directory, show_file_paths=False):
    file_counter = 0
    files_to_proceed = []
    for root, dirs, files in os.walk(directory):
        if len(files) > 0:
            for file in files:
                file_path = root + os.path.sep + file
                if file.endswith(".jpg"):
                    files_to_proceed.append(file_path)
                    file_counter += 1
                    if show_file_paths:
                        print(f"{file_counter}. {file_path}")

                # print(sum(getsize(join(root, name)) for name in files), end="")
                # print("bytes in", len(files), "non-directory files")
                # if 'CVS' in dirs:
                #     dirs.remove('CVS')  # don't visit CVS directories
                # for file in files:
                #     print(file)
                # for dir in dirs:
                #     print(dir)
    print(f"Znaleziono: {file_counter} plików .jpg")
