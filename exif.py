import subprocess

subprocess.call(f"exiftool -a -u -g1 out.jpg")
subprocess.call(f"exiftool -overwrite_original"
                # f" -Artist=Geotech"
                # f" -XPAuthor=Geotech"
                # f" -Copyright=Geotech"
                # f" -Rights=Geotech"
                # f" -Owner=Geotech"
                # 
                # f" -Software=GeotechApp"
                # f" -ProcessingSoftware=PythonOpenCV"
                # 
                # # f" -DateTimeOriginal=2021:08:13 13:13:13.13 +02:00\""
                # # f" -CreateDate=\"2021:01:01 01:01:01 +02:00\""
                # f" -AllDates=\"2021:06:06 06:06:06+06:00\""
                # 
                # f" -Title=tytu4"
                # f" -XPTitle=tytu4"
                # 
                # f" -XPSubject=temat"
                # 
                # f" -XPKeywords=\"tag9;tag9; t a g 9\""
                # 
                # f" -XPComment=komentarz"
                f" out.jpg")
subprocess.call(f"exiftool -a -u -g1 out.jpg")
