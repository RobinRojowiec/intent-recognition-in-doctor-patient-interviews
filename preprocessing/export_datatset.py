"""

IDE: PyCharm
Project: masterthesis
Author: Robin
Filename: export_datatset
Date: 25.11.2019

"""
import glob
import json
import os
import re
import shutil
from copy import deepcopy

from tqdm import tqdm

from common.paths import TRANSCRIPT_JSON_PATH, ROOT_RELATIVE_DIR, TRANSCRIPT_EXPORT_PATH, TRANSCRIPT_PATH


def translate_class_name(class_name):
    category = re.match("([A-Z]{2,6})", class_name).group(0)
    id = class_name[len(category):]

    new_category = category
    if category == "HS":
        new_category = "MS"
    elif category == "MV":
        new_category = "PH"
    elif category == "SUE":
        new_category = "SR"
    elif category == "NF":
        new_category = "IQ"
    elif category == "AF":
        new_category = "OQ"
    elif category == "AM" or category == "SF" or category == "OTHER":
        new_category = category
    else:
        print("Unknown category %s" % category)
    return new_category + id


export_files = []

print("Exporting files...")
for filename in tqdm(list(glob.iglob(ROOT_RELATIVE_DIR + TRANSCRIPT_JSON_PATH + '*.json', recursive=True))):
    with open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    data = deepcopy(data)
    del data[0]
    del data[1]
    del data[len(data) - 1]

    for item in data:
        item["classes"][0] = translate_class_name(item["classes"][0])

    export_filename = os.path.basename(filename)
    export_file = ROOT_RELATIVE_DIR + TRANSCRIPT_EXPORT_PATH + export_filename

    with open(export_file, "w+", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)
        export_files.append(export_file)

print("Creating archive...")
shutil.make_archive(ROOT_RELATIVE_DIR + TRANSCRIPT_PATH + 'export', 'zip',
                    os.path.abspath(ROOT_RELATIVE_DIR + TRANSCRIPT_EXPORT_PATH))

print("Done!")
