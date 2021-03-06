import re
import subprocess

xp_comment = "XPComment"
user_comment = "UserComment"
image_description = "ImageDescription"


def read_tag_value(tag, source_file, print_info=False):
    args = ["exiftool", "-s", "-s", "-s", f"-{tag}", f"{source_file}"]
    out, err = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True).communicate()
    if print_info:
        print(f"Exif read: {tag} = {out}")
        print("exif.read_tag_value:", out)
    return out.strip()


def write_tag_value(tag, value, source_file, print_info=False):
    write_args = ["exiftool", "-overwrite_original", f"-{tag}={value}", f"{source_file}"]
    out, err = subprocess.Popen(write_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True).communicate()
    if print_info:
        print(f"Exif write: {tag} = {value}")
        print("exif.write_tag_value:", out)


def write_resolution_tags(path_to_file, scale_in_dpmm, original_comment, print_info=False):
    if scale_in_dpmm < 0:
        return
    else:
        dpi = 25.4 * scale_in_dpmm
        formatted_scale = "{:.0f}".format(scale_in_dpmm)
        final_comment = f"{original_comment}calc{formatted_scale}dpmm;"
        args = f"exiftool -overwrite_original" \
               f" -XResolution={dpi}" \
               f" -YResolution={dpi}" \
               f" -ResolutionUnit=inches" \
               f" -ProcessingSoftware=PythonOpenCV" \
               f" -XPComment={final_comment}" \
               f" -UserComment={final_comment}" \
               f" {path_to_file}"
        out, err = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    universal_newlines=True).communicate()
        if print_info:
            print("exif.write_resolution_tags:", out)


def print_tag_value(tag, source_file, print_info=False):
    read_args = ["exiftool", f"-{tag}", f"{source_file}"]
    out, err = subprocess.Popen(read_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True).communicate()
    if print_info:
        print("exif.print_tag_value:", out)


def print_all_tags(source_file, print_info=False):
    args = f"exiftool -a -u -g1 {source_file}"
    out, err = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True).communicate()
    if print_info:
        print("exif.print_all_tags:", out)


def delete_all_tags(source_file, print_info=False):
    args = f"exiftool -all= {source_file}"
    out, err = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True).communicate()
    if print_info:
        print("exif.delete_all_tags:", out)


def copy_all_tags(source_file, destination_file, print_info=False):
    args = f"exiftool -overwrite_original -TagsFromFile {source_file} -all:all>all:all {destination_file}"
    out, err = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True).communicate()
    if print_info:
        print("exif.copy_all_tags:", out)


# def copy_tags_from_filename(destination_file, filename, print_info=False):
    # comment_tags = prepare_comment_tags(filename, print_info)
    # args = f"exiftool -overwrite_original" \
    #        f" -XPComment={comment_tags}" \
    #        f" -UserComment={comment_tags}" \
    #        f" {destination_file}"
    # out, err = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    #                             universal_newlines=True).communicate()
    # if print_info:
    #     print("exif.copy_tags_from_filename:", out)


def set_new_dpmm(dpmm, source_file, print_info=False):
    dpmm_pattern = r"\d+dpmm"
    new_dpmm = f"{dpmm}dpmm"
    original_tag = read_tag_value(xp_comment, source_file)
    if is_blank(original_tag):
        dpmm_replaced = new_dpmm
    elif re.search(dpmm_pattern, original_tag):
        dpmm_replaced = re.sub(dpmm_pattern, new_dpmm, original_tag)
    elif original_tag.endswith(";"):
        dpmm_replaced = original_tag + new_dpmm + ";"
    else:
        dpmm_replaced = ";" + new_dpmm + ";"
    if print_info:
        print("new_dpmm:", dpmm_replaced)
    write_tag_value(xp_comment, dpmm_replaced, source_file)


def is_blank(my_string):
    return not (my_string and my_string.strip())

# subprocess.call(f"exiftool -s -s -s -Artist out.jpg")
# subprocess.call(f"exiftool -overwrite_original"
#                 f" -Artist=Geotech"
#                 f" -XPAuthor=Geotech"
#                 f" -Copyright=Geotech"
#                 f" -Rights=Geotech"
#                 f" -Owner=Geotech"
#
#                 f" -Software=GeotechApp"
#                 f" -ProcessingSoftware=PythonOpenCV"
#
#                 # f" -DateTimeOriginal=2021:08:13 13:13:13.13 +02:00\""
#                 # f" -CreateDate=\"2021:01:01 01:01:01 +02:00\""
#                 f" -AllDates=\"2021:06:06 06:06:06+06:00\""
#
#                 f" -Title=tytu4"
#                 f" -XPTitle=tytu4"
#
#                 f" -XPSubject=temat"
#
#                 f" -XPKeywords=\"tag9;tag9; t a g 9\""
#
#                 f" -XPComment=komentarz"
#
#                 f" -GPSLatitude=\"53;53;53.53\""
#                 f" -GPSLatitudeRef=N"
#                 f" -GPSLongitude=\"18;18;18.18\""
#                 f" -GPSLongitudeRef=E"
#                 f" -GPSAltitude=123.123"
#                 f" -GPSAltitudeRef=0"  # above sea level
#                 f" -GPSDOP=012.012"  # Dilution Of Precision (geometric)
#
#                 f" out.jpg")
# subprocess.call(f"exiftool -a -u -g1 out.jpg")
