"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: visualize.py
Date: 20.07.2019

"""
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from common.paths import ROOT_RELATIVE_DIR, MODEL_PATH, DATA_PATH


def plot_log(file_name, name, ticks, key="loss"):
    epochs, train_loss, dev_loss = [], [], []
    with open(ROOT_RELATIVE_DIR + MODEL_PATH + file_name, encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            epochs.append(float(row[0]))
            train_loss.append((round(float(row[1]) * 100)) / 100.0 / 7.5)
            dev_loss.append((round(float(row[2]) * 100)) / 100.0)

        plt.yticks(ticks)
        plot_lines((epochs, train_loss, "train"), (epochs, dev_loss, "dev"))
        plt.savefig(ROOT_RELATIVE_DIR + DATA_PATH + "diagrams/" + name + ".png", format="png")
        plt.show()


def plot_lines(*lines):
    colors = ["green", "blue", "black"]
    for i, line in enumerate(lines):
        plot_line(line[0], line[1], line[2], color=colors[i])


def plot_line(values_x, values_y, label_x, **kwargs):
    plt.title("Loss over time")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(values_x, values_y, '-g', label=label_x, **kwargs)
    # plt.axis('equal')

    plt.legend()


def plot_histogram(values, name_x, name_y):
    sorted_values = list(sorted(values, key=lambda x: x, reverse=True))
    plt.bar(range(len(sorted_values)), sorted_values, width=0.5, align='center', color='darkgray', )

    plt.ylabel(name_y)
    plt.yticks((0, 2, 5, 10, 20, 30, 40, 50, 60))
    plt.ylim((0, 60))
    plt.xlabel(name_x)

    plt.title('Number of tokens per sample')
    plt.savefig(ROOT_RELATIVE_DIR + DATA_PATH + "diagrams/" + "tokens_per_sample" + ".png", format="png")
    plt.clf()


def plot_distribution(sizes, name_x, name_y, num_bins=10):
    maximum = 100
    sizes.sort(key=lambda x: x[0], reverse=True)
    sizes = [(min(item[0], maximum), item[1]) for item in sizes]
    size = [item[0] for item in sizes]
    labels = [item[1] for item in sizes]

    n, bins, patches = plt.hist(size, num_bins, facecolor='darkgray', align="mid", rwidth=0.9, alpha=0.5)

    plt.ylabel(name_y)
    plt.yticks((1, 2, 5, 10, 15, 20, 25))

    plt.xlabel(name_x)

    step = int(maximum / num_bins)
    bucket_range = range(0, maximum + step, step)
    plt.xticks(bucket_range)

    plt.title('Number of samples per class')
    plt.savefig(ROOT_RELATIVE_DIR + DATA_PATH + "diagrams/" + "samples_per_class" + ".png", format="png")
    plt.show()
    plt.clf()


# taken from: https://scikit-learn.org/stable/auto_examples/model_selection/
# plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes, model_name,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    class_list = list(set(classes))
    class_index = {class_name: i for i, class_name in enumerate(class_list)}
    class_labels = [class_name for class_name in classes]

    y_true = [class_index[class_name] for class_name in y_true]
    y_pred = [class_index[class_name] for class_name in y_pred]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # print out report
    print(classification_report(y_true, y_pred))

    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_labels, yticklabels=class_labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.set_size_inches(18.5, 10.5)

    plt.savefig(ROOT_RELATIVE_DIR + DATA_PATH + "diagrams/" + model_name + ".png", format="png")
    return ax


if __name__ == '__main__':
    classes = ["A", "B", "C"]
    real = ["A", "B", "B", "A", "A"]
    pred = ["C", "B", "A", "A", "A"]
    # plot_confusion_matrix(real, pred, classes)
    # plt.show()

    x, y = [1, 2, 3], [2, 3, 4]
    x2, y2 = [2, 2, 3], [2, 4, 4]
    # plot_lines( (x, y, "dev", "Accuracy"), (x2, y2, "test", "Accuracy"))
    # plot_log("BERTClassifier.csv")
    # plot_log("SiameseCNN.csv")
    # plot_log("bert_single/BERTWithConversationContext.csv", "Loss trend of BERT Single with highest accuracy",
    #        (10, 20, 30, 40))
    # plot_log("snn-cnn-sample/SiameseNeuralNetwork.csv", "Loss trend of SNN-CNN with highest accuracy", (0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0))

    objects = ('BERT+Trans. Probs', 'BM25', 'SNN-CNN')
    y_pos = np.arange(len(objects))
    performance = [71.88, 62.15, 54.69]
    performance2 = [65.47, 73.88, 61.98]

    fig, ax = plt.subplots()
    index = np.arange(3)
    bar_width = 0.3
    opacity = 0.8

    rects1 = plt.bar(index, performance, bar_width,
                     alpha=opacity,
                     color='darkgreen',
                     label='Accuracy', align='center', )

    rects2 = plt.bar(index + bar_width, performance2, bar_width,
                     alpha=opacity,
                     color='gray',
                     label='MAP', align='center', )

    plt.xlabel('Model')
    plt.ylabel('Scores')
    plt.title('Best scores by model')
    plt.xticks(index + bar_width, objects)
    plt.legend()
    plt.savefig(ROOT_RELATIVE_DIR + DATA_PATH + "diagrams/overall_scores.png", format="png")
    plt.show()
