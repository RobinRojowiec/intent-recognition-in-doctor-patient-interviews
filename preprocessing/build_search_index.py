"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: build_search_index
Date: 26.04.2019
Loads the transcripts in JSON format and stores them in a search index based on Whoosh
"""
from collections import defaultdict

from common.paths import SEARCH_INDEX_PATH, ROOT_RELATIVE_DIR
from utility.misc import convert_to_json_list
from utility.search_engine import SearchEngine


def build_search_index(path_train_set):
    def aggregate(utterances: []):
        aggregated = []
        data_set = defaultdict(lambda: dict({'utterance': [], 'classes': [], 'position': -1}))
        for utterance in utterances:
            for clazz in utterance['classes']:
                data_set[clazz]['utterance'].append(utterance['utterance'])
                data_set[clazz]['classes'] = [clazz]

        for key in data_set:
            utterance_set = data_set[key]
            utterance_set['utterance'] = ' '.join(utterance_set['utterance'])
            aggregated.append([utterance_set])

        return aggregated

    utterances = convert_to_json_list(path_train_set)
    aggregated_utterances = aggregate(utterances)

    # index
    engine: SearchEngine = SearchEngine(ROOT_RELATIVE_DIR + SEARCH_INDEX_PATH, read_only=False)
    engine.clear()
    engine.add_documents(aggregated_utterances)

    # quick check if index is fine
    assert len(engine.search('Trat das schonmal auf ?')) != 0
    print("Index building done!")
