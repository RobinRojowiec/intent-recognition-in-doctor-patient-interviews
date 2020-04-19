"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: analyse_data
Date: 26.04.2019

"""

import glob
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

import spacy
from nltk.corpus import stopwords
from pandas import DataFrame
from tqdm import tqdm

from common.paths import TRANSCRIPT_JSON_PATH, ROOT_RELATIVE_DIR

tokenizer_de = spacy.load('de_core_news_sm').tokenizer
stopwords_de = set(stopwords.words('german'))


def get_tokens(sent):
    return [token.text for token in tokenizer_de(sent)]


clusters: dict = defaultdict(list)
category_clusters = defaultdict(int)

vocab = set()
max_len = 0
min_len = sys.maxsize
sample_sizes = []
sample_counter = 0
conv_sizes = []
low_sample_counter = 0
multi_class_samples = 0

for filename in tqdm(list(glob.iglob(ROOT_RELATIVE_DIR + TRANSCRIPT_JSON_PATH + '*.json', recursive=True))):
    with open(filename, encoding="utf-8") as json_file:
        data = json.load(json_file)
    transcript_frame: DataFrame = DataFrame(data)
    conv_sizes.append(len(transcript_frame))

    if len(transcript_frame) < 4:
        os.remove(filename)
        continue

    if len(transcript_frame) > 400:
        print("Shorty: " + os.path.basename(filename))
        for index, row in transcript_frame.iterrows():
            print(row["utterance"] + "\t %s \t" % ",".join(row["classes"]))
        # print(transcript_frame)

    for index, row in transcript_frame.iterrows():
        if len(row["classes"]) > 1:
            multi_class_samples += 1
            print("\n", row["utterance"], " - ", row["classes"])
        for clazz in row["classes"]:
            utterance = row["utterance"].replace("\"", "")
            if "Alkohol" in utterance:
                print("Found %s" % utterance)
            clusters[clazz].append(utterance)
            sample_counter += 1
            tokens = get_tokens(utterance)

            max_len = max(max_len, len(tokens))
            min_len = min(min_len, len(tokens))

            if len(tokens) <= 10:
                low_sample_counter += 1

            sample_sizes.append(len(tokens))

            for token in tokens:
                vocab.add(token)

        for symptom_class in row["classes"]:
            category = re.split("[0-9]+", symptom_class)[0]
            category_clusters[category] += 1

few_sample_counter = 0
few_sample_limit = 3
for k, v in clusters.items():
    if len(v) < few_sample_limit:
        few_sample_counter += 1
    print("%s : %i" % (k, len(v)))

print("\n")
conv_sizes.sort()
print(conv_sizes)
print("Number of conversations %s" % len(conv_sizes))
print("Mean conversation length %s " % statistics.mean(conv_sizes))
print("Max conversation length %s " % max(conv_sizes))
print("Min conversation length %s " % min(conv_sizes))

mean_all = statistics.mean(sample_sizes)
print("Mean %.02f" % mean_all)
mode_all = statistics.mode(sample_sizes)
print("Mode %.02f" % mode_all)
standart_dev = statistics.stdev(sample_sizes, xbar=mean_all)
print("Standart dev: % s" % standart_dev)

sample_sizes.sort()
print(sample_sizes)

print("Multi class samples: %i" % multi_class_samples)
print("Classes with less than %i samples: %i" % (few_sample_limit, few_sample_counter))
print("Number of classes: %i" % len(clusters))
print("Fraction of low sample classes: %2.2f " % (few_sample_counter / (float(len(clusters)))))
print("Done")

print("Number of samples: %i" % sample_counter)
print("Vocab size: %i" % len(vocab))
print("Vocab size (without stopwords): %i" % len(list(filter(lambda x: x.lower() not in stopwords_de, vocab))))
print("Max len %i" % max_len)
print("Min len %i" % min_len)
print("Low sample size: %i" % low_sample_counter)

for k, v in category_clusters.items():
    print("%s : %.02f " % (k, (v + 0.0) / sample_counter * 100))

# examples
top_classes = []
top_items_count = 10
for k, v in clusters.items():
    top_classes.append((k, len(v), v[random.randint(0, len(v) - 1)]))

top_classes.sort(reverse=True, key=lambda x: x[1])
top_classes = top_classes[:top_items_count]

print("\hline")
for top_class in top_classes:
    print("%s & %i & %s\\\\\n\hline " % (top_class[0], top_class[1], top_class[2]))

# plot_histogram([nof[1] for nof in top_classes],"Classes", "Number of samples", title="Number of samples for top 20 classes", width=0.6)

# plot_histogram(conv_sizes, "Number of conversations", "Number of utterances",width=0.6)
# plot_distribution([(val, str("val")) for val in conv_sizes], 'Number of samples', 'Number of classes', title="Number of utterances per dialogue")

# plot_histogram(sample_sizes, "Number of samples", "Size of sample in tokens")
# plot_distribution([(len(clusters[key]), key) for key in clusters], 'Number of samples', 'Number of classes')
