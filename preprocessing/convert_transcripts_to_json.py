"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: convert_transcripts_to_json
Date: 26.04.2019

"""

import glob
import json
import os
import re

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from common.paths import TRANSCRIPT_ORIGINAL_PATH, TRANSCRIPT_JSON_PATH

OTHER_CLASS = "OTHER"
class_split_pattern = re.compile("[A-Z]*")
column_split_pattern = re.compile("[,;]")


def translate_codes_to_ids(codes: str):
    """
    translate a list of utterance classes to class ids
    :param codes:
    :return:
    """
    code_splits = column_split_pattern.split(codes)
    classes = []
    for code in code_splits:
        classes.extend(translate_code_to_id(code))
    return list(filter(lambda x: x is not None and x != '', classes))


def translate_code_to_id(code: str):
    """
    Checks and translates code1, code2 IDs
    :param code:
    :return:
    """
    code: str = code.strip()

    if code in ["NF", "AF"]:
        return [OTHER_CLASS]
    else:
        return [code]


def json_entry(index: int, utterance: str, classes: [], previous_classes: [], previous_utterance: str):
    """
    Generates a valid utterance json
    :param index:
    :param utterance:
    :param classes:
    :param previous_classes:
    :param previous_utterance:
    :return:
    """
    expression_json: dict = {
        "position": index,
        "utterance": utterance,
        "classes": classes,
        "previous_classes": previous_classes,
        "previous_utterance": previous_utterance
    }
    return expression_json


def save_json(entries: list, filename: str, part: int):
    """
    stores the json on disk
    :param entries:
    :param filename:
    :param index:
    :return:
    """
    filename_code: str = filename.split("_")[1][:7]
    full_filename: str = '../' + TRANSCRIPT_JSON_PATH + filename_code + "_" + str(part) + ".json"

    if os.path.exists(full_filename):
        os.remove(full_filename)
    with open(full_filename, encoding="utf-8", mode="w+") as json_file:
        json.dump(entries, json_file, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    print("Generating JSONs from spreadsheets...")
    for filename in tqdm(list(glob.iglob('../' + TRANSCRIPT_ORIGINAL_PATH + '*.xlsx', recursive=True))):
        transcript_frame: DataFrame = pd.read_excel(filename)

        # indicies: 3=>Text, 4=>Code1, 5=>Code2
        print('\n', "Processing file %s" % filename)

        entries: [] = []
        position = 0
        part = 1
        previous_classes = ["START"]
        previous_utterance = ""
        for index, row in transcript_frame.iterrows():
            code1: str = transcript_frame.iloc[index, 4]

            if code1 in ["PBF", "BEZ", '']:
                utterance, code2 = transcript_frame.iloc[index, 3], transcript_frame.iloc[index, 5]
                position += 1

                if code1 == "PBF":
                    classes = translate_codes_to_ids(code2)
                else:
                    classes = [OTHER_CLASS]

                entry = json_entry(position, utterance, classes, previous_classes, previous_utterance)
                previous_classes = classes
                previous_utterance = utterance

                if not isinstance(utterance, str) or '' in classes:
                    if len(entries) == 0:
                        break
                    save_json(entries, filename, part)
                    entries = []
                    position = 0
                    part += 1
                else:
                    entries.append(entry)
        save_json(entries, filename, part)
