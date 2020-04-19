"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: testing.py
Date: 16.07.2019

"""
import csv

import scipy
import torch
from sklearn.metrics import classification_report
from torchtext.data import TabularDataset, BucketIterator, Field
from tqdm import tqdm

from common.paths import ROOT_RELATIVE_DIR, MODEL_PATH
from preprocessing.generate_data_sets import NEGATIVE_SAMPLE_SIZE
from utility.evaluation import _calculate_map, build_relevance_list
from utility.model_parameter import Configuration, ModelParameter

torch.cuda.current_device()


def is_correct(true_label, pred_label):
    if true_label == pred_label:
        return 1
    return 0


def get_significance(true_labels: [], system_a_preds: [], system_b_preds: []):
    differ = 0
    a_better_b = 0
    for index, label in enumerate(true_labels):
        system_a_pred = system_a_preds[index]
        system_b_pred = system_b_preds[index]

        correct_a, correct_b = is_correct(label, system_a_pred), is_correct(label, system_b_pred)
        if correct_a != correct_b:
            differ += 1
        if correct_a > correct_b:
            a_better_b += 1

    return scipy.stats.binom_test(differ, a_better_b)


def get_loss(model, batch, task, criterion, device, margin=0.2, mode="training"):
    real_classes = batch.sample_class
    negative_samples = [getattr(batch, "sample_neg_" + str(i)) for i in range(NEGATIVE_SAMPLE_SIZE)]

    if task == "ranking":
        scores = []
        scores.append((model.compare(batch.sample, batch.sample_pos), 1))
        for negative_sample in negative_samples:
            sim = model.compare(batch.sample, negative_sample, device=device, mode="eval")
            scores.append((sim, 0))

        pos, neg = model(batch.sample, batch.previous_classes, batch.position, batch.previous_sample,
                         batch.sample_pos, *negative_samples, device=device, mode="eval")
        loss = criterion(pos[0], neg[0], margin)
        predicted_classes = torch.cat((torch.unsqueeze(pos, 1), torch.unsqueeze(neg, 1)), 1)
        class_probs = torch.cat((torch.unsqueeze(pos, 1), torch.unsqueeze(neg, 1)), 0)[0]
        return loss, [predicted_classes, class_probs, scores]

    else:
        predicted_classes, class_probs = model(batch.sample, batch.previous_classes, batch.position,
                                               batch.previous_sample,
                                               batch.sample_pos, *negative_samples, device=device)
        loss = criterion(predicted_classes, real_classes)

        return loss, [predicted_classes, class_probs]


def class_map(predicted_classes, real_classes):
    map_score = .0
    number_of_samples = real_classes.size()[0]
    for i in range(number_of_samples):
        real_class_index = [real_classes[i].item()]

        predictions = [(pred_index, pred_score.item()) for pred_index, pred_score in enumerate(predicted_classes[i])]
        predictions.sort(key=lambda x: x[1], reverse=True)
        prediction_indices = [pred[0] for pred in predictions]

        map_score += _calculate_map(build_relevance_list(real_class_index, prediction_indices))

    return map_score / number_of_samples


def accuracy(predicted_classes, real_classes):
    acc_score = 0.0
    number_of_samples = real_classes.size()[0]
    for i in range(real_classes.size()[0]):
        if real_classes[i] == torch.argmax(predicted_classes[i]):
            acc_score += 1
    return acc_score / number_of_samples


def rank_accuracy(scores, batch_size, k=1):
    if len(scores) == 0:
        return .0
    prec_scores = .0
    for i in range(batch_size):
        scores.sort(key=lambda x: x[0][i], reverse=True)
        relevance_list = [sims[1] for sims in scores]
        prec_scores += 1 if 1 in relevance_list[:k] else 0
    return prec_scores / batch_size


def rank_map(scores, batch_size):
    if len(scores) == 0:
        return .0
    map_scores = .0
    for i in range(batch_size):
        scores.sort(key=lambda x: x[0][i], reverse=True)
        relevance_list = [sims[1] for sims in scores]
        map_scores += _calculate_map(relevance_list)
    return map_scores / batch_size


def evaluate_epoch(model, iterator, criterion, mode="Validation", batch_size=32, device='cpu', task="classification",
                   margin=0.2, classes=[]):
    """
    Evaluate model predictions
    :param model:
    :param iterator:
    :return:
    """
    acc_score, total, val_loss, map_score = 0.0, 0, 0.0, .0
    true_classes = []
    predicted_classes = []
    with torch.no_grad():
        print("Evaluating %s Set with %i samples" % (mode, len(iterator) * batch_size), "\n")
        for batch in tqdm(iterator):
            model.eval()
            real_classes = batch.sample_class
            loss, outputs = get_loss(model, batch, task, criterion, device, margin=margin)
            val_loss += loss.item()

            if task == "classification":
                acc_score += accuracy(outputs[0], real_classes)
                map_score += class_map(outputs[1], real_classes)
            else:
                map_score += rank_map(outputs[2], batch.sample.size()[0])
                acc_score += rank_accuracy(outputs[2], batch.sample.size()[0])

            total += 1

            predicted_classes += [torch.argmax(outputs[0][i]).item() for i in range(outputs[0].size()[0])]
            true_classes += [real_classes[i].item() for i in range(real_classes.size()[0])]

    if mode == 'Test':
        print(classification_report(true_classes, predicted_classes, output_dict=False, digits=4))
        with open(ROOT_RELATIVE_DIR + MODEL_PATH + "classifications.csv", 'w+', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(predicted_classes)):
                writer.writerow([true_classes[i], predicted_classes[i]])

    print("\n")
    print(mode, 'Loss: %.02f' % val_loss)
    print(mode, 'Accuracy: %.02f' % (100 * acc_score / total), "%")
    print(mode, 'MAP: %.02f' % (100 * map_score / total), "%")
    return acc_score / total, map_score / total, val_loss


def evaluate_model(model, criterion, device, config: Configuration, data_fields, task="classification", margin=0.2,
                   shuffle=True):
    batch_size = config.get_int(ModelParameter.BATCH_SIZE)
    test_data_file = config.get_string(ModelParameter.TEST_FILE)

    class_field: Field = data_fields[0][1]
    classes = [cls for cls in class_field.vocab.stoi][1:]

    # load and prepare test dataset
    test_data = TabularDataset(path=test_data_file,
                               format='tsv',
                               fields=data_fields, skip_header=True)

    test_iterator = BucketIterator(test_data,
                                   batch_size=batch_size,
                                   device=device,
                                   sort=True,
                                   sort_key=lambda x: x.sample,
                                   sort_within_batch=True,
                                   shuffle=shuffle
                                   )

    return evaluate_epoch(model, test_iterator, criterion, mode="Test", batch_size=batch_size, device=device, task=task,
                          margin=margin, classes=classes)


if __name__ == '__main__':
    true_labels = [2, 33, 15, 16, 17]
    system_a = [2, 30, 16, 16, 15]
    system_b = [1, 10, 13, 15, 16]
    print(scipy.stats.binom_test(7, 8))
    print(get_significance(true_labels, system_a, system_b))
