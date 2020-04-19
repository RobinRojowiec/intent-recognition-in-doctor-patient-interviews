"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: retrieval
Date: 26.04.2019

"""

import csv
import pickle

from common.paths import ROOT_RELATIVE_DIR, PREPROCESSED_TRANSCRIPT_PATH, MODEL_PATH
from probability.tables import TransitionTable


def count_and_store_transitions():
    with open(ROOT_RELATIVE_DIR + PREPROCESSED_TRANSCRIPT_PATH + "class_samples_train.tsv", encoding='utf-8',
              newline='') as transcript_file:
        reader = csv.reader(transcript_file, delimiter='\t')

        # collect data
        state_names = set()

        # skip header
        next(reader)

        transitions: [] = []
        for index, row in enumerate(reader):
            clazz = row[0]
            previous_class = row[3]

            transitions.append([previous_class, clazz])

            state_names.add(clazz)
            state_names.add(previous_class)

        # build and test probability table
        tt = TransitionTable(list(state_names))
        for transition in transitions:
            tt.count_transition(transition[0], transition[1])

        with open(ROOT_RELATIVE_DIR + MODEL_PATH + "transition_table.pckl", "wb") as file_prt:
            pickle.dump(tt, file_prt)
        for row in tt.data:
            print(row)


if __name__ == '__main__':
    count_and_store_transitions()
