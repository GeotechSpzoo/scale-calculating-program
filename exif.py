import re
import subprocess

xp_comment = "XPComment"
user_comment = "UserComment"


def read_tag_value(tag, source_file):
    args = ["exiftool", "-s", "-s", "-s", f"-{tag}", f"{source_file}"]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = process.communicate()
    # print(f"Exif read: {tag} = {out}")
    return out


def write_tag_value(tag, value, source_file):
    write_args = ["exiftool", "-overwrite_original", f"-{tag}={value}", f"{source_file}"]
    print(f"Exif write: {tag} = {value}")
    subprocess.Popen(write_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    # print_tag_value(tag, source_file)


def print_tag_value(tag, source_file):
    read_args = ["exiftool", f"-{tag}", f"{source_file}"]
    subprocess.Popen(read_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def print_all_tags(source_file):
    args = f"exiftool -a -u -g1 {source_file}"
    subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def delete_all_tags(source_file):
    args = f"exiftool -all= {source_file}"
    subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def set_new_dpmm(dpmm, source_file):
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
    print("new_dpmm:", dpmm_replaced)
    write_tag_value(xp_comment, dpmm_replaced, source_file)


def is_blank(my_string):
    return not (my_string and my_string.strip())


def copy_all_tags(source_file, destination_file):
    args = f"exiftool -q -q -overwrite_original -TagsFromFile {source_file} -all:all>all:all {destination_file}"
    subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

# write_tag_value(xp_comment, "0;graw01;0m;DRY;zoom-in;IR;706dpmm;")
# set_new_dpmm(123)

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
