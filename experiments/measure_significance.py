"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: measure_significance.py
Date: 28.07.2019

"""
import csv

from common.paths import ROOT_RELATIVE_DIR, MODEL_PATH
from utility.testing import get_significance


def load_classifications(name):
    labels, pred = [], []
    with open(ROOT_RELATIVE_DIR + MODEL_PATH + name, 'w+', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(row[0])
            pred.append(row[1])
    return labels, pred


system_a = 'classifications_bert_single.csv'
system_b = 'classifications_bert_probs.csv'
true_labels = []

true_labels, system_a_cls = load_classifications(system_a)
true_labels, system_b_cls = load_classifications(system_b)

print("Significance: %.04f" % get_significance(true_labels, system_b, system_a))
