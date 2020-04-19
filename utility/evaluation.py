"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: test
Date: 26.04.2019

"""

from copy import deepcopy


def build_relevance_list(real_labels: [], predicted_labels: []):
    real_labels_copy = deepcopy(real_labels)

    relevance_list: [] = []
    for predicted_label in predicted_labels:
        if predicted_label in real_labels_copy:
            relevance_list.append(1)
            real_labels_copy.remove(predicted_label)
        else:
            relevance_list.append(0)
    return relevance_list


def calculate_rr(real_labels: [], predicted_labels: []):
    relevance_list = build_relevance_list(real_labels, predicted_labels)
    for index, item in enumerate(relevance_list):
        if item == 1:
            return 1 / (index + 1)
    return 0.0


def calculate_recall(real_labels: [], predicted_labels: []):
    recall: float = 0.0
    for real_label in real_labels:
        if real_label in predicted_labels:
            recall += 1.0
    return recall / len(real_labels)


def calculate_accuracy(real_labels: [], predicted_labels: []):
    correct: float = 0
    for label in predicted_labels:
        if label in real_labels:
            correct += 1.0

    return correct / len(predicted_labels)


def calculate_avp(real_labels: [], predicted_labels: []):
    relevance_list = build_relevance_list(real_labels, predicted_labels)

    avp: float = 0
    counter: float = 0

    for i, element in enumerate(relevance_list):
        if element == 1:
            prec = sum(relevance_list[:i + 1])
            avp += prec / (i + 1)
            counter += 1

    if counter == 0:
        return 0
    return avp / counter


def calculate_map(list_real_labels: [], list_predicted_labels: []):
    map: float = .0
    instances = len(list_real_labels)

    for i in range(len(list_real_labels)):
        map += calculate_avp(list_real_labels[i], list_predicted_labels[i])

    return map / instances


def _calculate_map(relevance_list: []):
    map: float = .0
    counter: float = 0
    last_relevant_index = 0

    for i, element in enumerate(relevance_list):
        if element == 1:
            prec = sum(relevance_list[last_relevant_index:i + 1])
            map += prec / (i + 1)
            counter += 1
            last_relevant_index = i

    return map / counter


if __name__ == '__main__':
    real_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    predicted_labels = ["Q1", "Q6", "Q3", "Q7", "Q5"]

    avp = calculate_avp(real_labels, predicted_labels)
    map = calculate_map([real_labels], [predicted_labels])
    acc = calculate_accuracy(real_labels, predicted_labels)

    print("AVP: %f" % avp)
    print("MAP: %f" % map)
    print("Accuracy: %f" % acc)
