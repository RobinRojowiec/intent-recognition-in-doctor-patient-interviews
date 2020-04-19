"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: ranking
Date: 26.04.2019

"""
from probability.tables import TransitionTable


class Ranker:
    def __init__(self):
        pass

    def calculate_score(self, result: {}):
        raise NotImplementedError

    def rank(self, results: []):
        scored_list: [] = []
        for result in results:
            new_score: float = self.calculate_score(result)
            result['score'] = new_score

            scored_list.append(result)
        return sorted(scored_list, key=lambda result: result['score'], reverse=True)


class TransitionRanker(Ranker):
    def __init__(self, transition_table):
        Ranker.__init__(self)
        self.transition_table: TransitionTable = transition_table

    def calculate_score(self, result: {}, state_a):
        if len(result['classes']) == 0:
            state_b = "INTRO"
        else:
            state_b = result['classes'][0]

        score: float = result['score']
        trans_prob: float = self.transition_table.get_probability(state_a, state_b)
        return score * trans_prob

    def rank(self, results: [], previous_states):
        scored_list: [] = []
        for result in results:
            # add probability probability
            result['score'] = max(self.calculate_score(result, state_a) for state_a in previous_states)
            scored_list.append(result)
        return sorted(scored_list, key=lambda result: result['score'], reverse=True)


class EmptyRanker(Ranker):
    def __init__(self):
        Ranker.__init__(self)

    def calculate_score(self, result: {}):
        return result['score']


class NeuralNetworkRanker(Ranker):
    def __init__(self):
        Ranker.__init__(self)

    def calculate_score(self, result: {}):
        return result['score']
