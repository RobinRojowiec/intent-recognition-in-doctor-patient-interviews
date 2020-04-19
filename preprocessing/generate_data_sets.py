import csv
import glob
import json
import random
from collections import defaultdict
from copy import deepcopy

from tqdm import tqdm

from common.paths import PREPROCESSED_TRANSCRIPT_PATH, TRANSCRIPT_JSON_PATH
from preprocessing.build_search_index import build_search_index
from preprocessing.estimate_probability_tables import count_and_store_transitions

# define paths
path_train_set = '../' + PREPROCESSED_TRANSCRIPT_PATH + "class_samples_train.tsv"
path_dev_set = '../' + PREPROCESSED_TRANSCRIPT_PATH + "class_samples_dev.tsv"
path_test_set = '../' + PREPROCESSED_TRANSCRIPT_PATH + "class_samples_test.tsv"

# define negative samples for ranking
NEGATIVE_SAMPLE_SIZE = 19

# set seed for reproducibility
random.seed(1234)


def generate_training_and_test_set(clusters):
    """
    Generates question cluster split for train/dev/test and saves it
    The split is stratified, meaning each class should have a similar distribution
    in each data set
    :param clusters:
    :return:
    """
    # splitting into training, development and test set
    train_ratio = 0.80
    dev_ratio = 0.10

    train_clusters = defaultdict(list)
    dev_clusters = defaultdict(list)
    test_clusters = defaultdict(list)

    for key in clusters.keys():
        questions = clusters[key]

        # ensure random order
        random.shuffle(questions)

        # calculate splits
        questions_size = len(questions)
        questions_size_train = max(int(questions_size * train_ratio), 1)
        questions_size_dev = max(round(questions_size * dev_ratio), 0)
        questions_size_test = max(questions_size - (questions_size_train + questions_size_dev), 0)

        # distribute samples to train/dev/test clusters
        for _ in range(questions_size_train):
            train_clusters[key].append(questions.pop())

        for _ in range(questions_size_dev):
            dev_clusters[key].append(questions.pop())

        for _ in range(questions_size_test):
            test_clusters[key].append(questions.pop())

    # save questions in csv format
    generate_csv(train_clusters, path_train_set, negative_sample_size=NEGATIVE_SAMPLE_SIZE)
    generate_csv(dev_clusters, path_dev_set, negative_sample_size=NEGATIVE_SAMPLE_SIZE)
    generate_csv(test_clusters, path_test_set, negative_sample_size=NEGATIVE_SAMPLE_SIZE)


def generate_csv(clusters, path_name, negative_sample_size=50):
    all_cluster_key_list = list(clusters.keys())
    sample_list = [
        ["sample_class", "sample", "position", "previous_class", "previous_utterance", "sample_pos"] + [
            "sample_neg_" + str(i) for i in
            range(negative_sample_size)]]

    # create samples for all clusters
    for question_class in all_cluster_key_list:
        # try to generate equally distributed positive/negative samples
        # if a class has less samples than negative_sample_size, the same sample
        # is added multiple times to avoid class bias
        current_cluster = clusters[question_class]
        for utterance in current_cluster:
            utterance_class = utterance["classes"][0]

            # collect a positive sample
            positive_sample = [random.choice(current_cluster)["utterance"]]
            negative_samples = []

            i = 0
            while i < negative_sample_size:
                # ensure a different class for sample selection
                while True:
                    random_question_class = random.choice(all_cluster_key_list)
                    if utterance_class != random_question_class:
                        break

                # ensure unique samples in the negative sample list
                random_cluster_size = len(clusters[random_question_class])
                included = 0
                while True:
                    random_utterance = random.choice(clusters[random_question_class])['utterance']
                    if random_utterance not in negative_samples:
                        negative_samples.append(random_utterance)
                        break
                    else:
                        included += 1
                    if random_cluster_size == included:
                        i -= 1
                        break
                i += 1
            sample_list.append([utterance_class, utterance["utterance"], utterance["position"],
                                utterance["previous_classes"][0],
                                utterance["previous_utterance"]] + positive_sample + negative_samples)

    print("Number of utterances: %i" % len(sample_list))
    with open(path_name, "w+", encoding="utf-8",
              newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter="\t", quotechar="\"", quoting=csv.QUOTE_ALL)
        for row in sample_list:
            csv_writer.writerow(row)


def down_sample_cluster(clusters, max_size):
    """
    Removes samples form clusters
    :param clusters:
    :param max_size:
    :return:
    """
    all_cluster_key_list = list(clusters.keys())
    for key in all_cluster_key_list:
        current_cluster = clusters[key]
        cluster_size = len(current_cluster)

        down_sample_size = max(0, cluster_size - max_size)
        random.shuffle(current_cluster)

        for i in range(down_sample_size):
            current_cluster.pop()


def filter_small_cluster(clusters, min_size=10):
    all_cluster_key_list = list(clusters.keys())
    for key in all_cluster_key_list:
        current_cluster = clusters[key]
        cluster_size = len(current_cluster)

        if cluster_size < min_size:
            del clusters[key]


if __name__ == '__main__':
    utterance_clusters = defaultdict(list)
    max_cluster_size = 100

    # collect utterances per class for stratified splits
    for filename in tqdm(list(glob.iglob('../' + TRANSCRIPT_JSON_PATH + '*.json', recursive=True))):
        with open(filename, 'r', newline='', encoding='utf-8') as json_file:
            utterance_list = json.load(json_file)
            for utterance in utterance_list:
                for clazz in utterance["classes"]:
                    utterance_copy = deepcopy(utterance)
                    utterance_copy["classes"] = [clazz]
                    for pre_class in utterance["previous_classes"]:
                        utterance_copy["previous_classes"] = [pre_class]
                        utterance_clusters[clazz].append(utterance_copy)

    # remove samples from to big clusters to equalize distributions
    down_sample_cluster(utterance_clusters, max_cluster_size)

    # display found classes
    class_list = [key for key in utterance_clusters]
    sizes = [len(utterance_clusters[key]) for key in utterance_clusters]
    print(class_list)
    print(sizes)
    print(len(class_list))

    # distribute the samples to sets and save them as tsv
    generate_training_and_test_set(utterance_clusters)

    # count transitions
    count_and_store_transitions()

    # load data into search index
    build_search_index(path_train_set)
