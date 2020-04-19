"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: misc.py
Date: 16.07.2019

"""
import csv

import torch

from preprocessing.convert_transcripts_to_json import json_entry


def id_vectors(samples, device='cpu'):
    """
    :param device:
    :param samples:
    :return:
    """
    return torch.cat([torch.LongTensor([[si for si in s]]) for s in samples]).to(device)


def convert_to_json_list(path):
    utterances = []
    with open(path, 'r', encoding='utf-8', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter="\t", )
        next(reader)
        for line in reader:
            utterance = json_entry(line[2], line[1], [line[0]], [line[3]], None)
            utterances.append(utterance)
    return utterances


def get_device():
    # check cuda support
    CUDA_SUPPORT = False
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        CUDA_SUPPORT = True
    else:
        device = torch.device("cpu")

    # device = torch.device('cpu')
    print("Device: %s" % str(device))
    return device, CUDA_SUPPORT
