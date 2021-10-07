import re
import subprocess

xp_comment = "XPComment"


def read_tag_value(tag):
    args = ["exiftool", "-s", "-s", "-s", f"-{tag}", "out.jpg"]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = process.communicate()
    print(f"Exif read: {tag} = {out}")
    return out


def write_tag_value(tag, value):
    write_args = ["exiftool", "-overwrite_original", f"-{tag}={value}", "out.jpg"]
    subprocess.call(write_args)
    print_tag_value(tag)


def print_tag_value(tag):
    read_args = ["exiftool", f"-{tag}", "out.jpg"]
    subprocess.call(read_args)


def print_all_tags():
    subprocess.call(f"exiftool -a -u -g1 out.jpg")


def set_new_dpmm(dpmm):
    dpmm_pattern = r"\d+dpmm"
    new_dpmm = f"{dpmm}dpmm"
    original_tag = read_tag_value(xp_comment)
    if is_blank(original_tag):
        dpmm_replaced = new_dpmm
    elif re.search(dpmm_pattern, original_tag):
        dpmm_replaced = re.sub(dpmm_pattern, new_dpmm, original_tag)
    elif original_tag.endswith(";"):
        dpmm_replaced = original_tag + new_dpmm + ";"
    else:
        dpmm_replaced = ";" + new_dpmm + ";"
    print("new_dpmm:", dpmm_replaced)
    write_tag_value(xp_comment, dpmm_replaced)


def is_blank(my_string):
    return not (my_string and my_string.strip())

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
