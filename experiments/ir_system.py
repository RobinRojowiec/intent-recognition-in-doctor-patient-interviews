"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: experiments
Date: 26.04.2019

"""
import pickle

from whoosh.scoring import BM25F, TF_IDF

from common.paths import SEARCH_INDEX_PATH, MODEL_PATH, ROOT_RELATIVE_DIR
from preprocessing.generate_data_sets import path_dev_set, path_test_set
from probability.tables import TransitionTable
from utility.evaluation import calculate_avp, calculate_recall, calculate_accuracy, calculate_rr
from utility.misc import convert_to_json_list
from utility.ranking import Ranker, TransitionRanker, EmptyRanker
from utility.search_engine import SearchEngine

# define experiment parameters

RESULT_LIMIT = 128
RETURN_RANGE = 1

# load corpus
search_engine: SearchEngine = SearchEngine(ROOT_RELATIVE_DIR + SEARCH_INDEX_PATH, read_only=True)

# load probability table and neural_bert_models
with open(ROOT_RELATIVE_DIR + MODEL_PATH + "transition_table.pckl", "rb") as file_prt:
    tt: TransitionTable = pickle.load(file_prt)


def do_search(scoring_function, transcripts, lambda_value=600, verbose=True):
    tt.lambda_value = lambda_value

    ranker_trans: Ranker = TransitionRanker(tt)
    ranker_empty: Ranker = EmptyRanker()

    counter = .0
    map_bm25f, mrr_bm25f, recall_bm25f, precision_bm25f = .0, .0, .0, .0
    map_trans, precision_trans, mrr_trans, recall_trans = .0, .0, .0, .0
    real_label, predicted_label = [], []

    for utterance in transcripts:
        query = utterance['utterance']
        real_classes: [] = utterance['classes']

        results = search_engine.search(query, limit=RESULT_LIMIT, scoring_function=scoring_function)
        if len(results) > 0:
            # statistical ranking
            unranked_results = ranker_empty.rank(results)
            pred_labels = []
            for result in unranked_results:
                pred_labels.extend(result['classes'])
            vsm_avp = calculate_avp([label for label in utterance['classes']], pred_labels)
            map_bm25f += vsm_avp
            mrr_bm25f += calculate_rr(real_classes, pred_labels)
            precision_bm25f += calculate_accuracy(real_classes, pred_labels[:RETURN_RANGE])
            recall_bm25f += calculate_recall(real_classes, pred_labels)

            # ranking using probability probabilities
            previous_classes = utterance['previous_classes']
            ranked_results = ranker_trans.rank(results, previous_classes)
            pred_labels = []
            for result in ranked_results:
                pred_labels.extend(result['classes'])
            avp = calculate_avp([label for label in utterance['classes']], pred_labels)
            map_trans += avp
            precision_trans += calculate_accuracy(real_classes, pred_labels[:RETURN_RANGE])
            mrr_trans += calculate_rr(real_classes, pred_labels)
            recall_trans += calculate_recall(real_classes, pred_labels)

            real_label.append(real_classes[0])
            predicted_label.append(pred_labels[0])

            counter += 1
        else:
            print("Empty results for: %s" % query)

    scoring_function_name = str(scoring_function.__name__)

    # calculate metrics
    map_score = (map_bm25f / counter)
    mrr_score = (mrr_bm25f / counter)
    precision_score = precision_bm25f / counter
    map_score_probs = (map_trans / counter)
    mrr_score_probs = (mrr_trans / counter)
    precision_score_probs = precision_trans / counter

    if verbose:
        print('\n')
        print("All runs: %i" % counter)
        # print("----------------------------------------------------------------------------")
        # print(scoring_function_name)
        # print("MAP: %f" % map_score)
        # print("MRR: %f" % mrr_score)
        # print("Precision@%i: %f" % (RETURN_RANGE, precision_score))
        print("----------------------------------------------------------------------------")
        print("%s + Trans Probs" % scoring_function_name)
        print("MAP: %f" % map_score_probs)
        # print("MRR: %f" % mrr_score_probs)
        print("Precision@%i: %f" % (RETURN_RANGE, precision_score_probs))

    return [scoring_function, lambda_value, map_score, mrr_score, precision_score, map_score_probs,
            mrr_score_probs, precision_score_probs], real_label, predicted_label, pred_labels


def evaluate(sf, dev_transcripts, lambdas, verbose=False):
    all_results = [("scoring_function", "lambda", "map", "mrr", "precision_at_" + str(RETURN_RANGE), "map_probs",
                    "mrr_probs", "precision_at_" + str(RETURN_RANGE) + "_probs")]

    best_score = 0.0
    best_config = None

    for lambda_value in lambdas:
        results, _, _, _ = do_search(sf, dev_transcripts, lambda_value, verbose=verbose)
        if verbose:
            print('\n')
            print("using lambda %.2f" % lambda_value)
        all_results.append(results)

        best_score_run = max(results[2], results[5])
        if best_score_run > best_score:
            best_score = best_score_run
            best_config = results

    return best_config


if __name__ == '__main__':
    # load dev data
    dev_transcripts = convert_to_json_list(path_dev_set)

    # define variables
    lambdas = [10, 50, 100, 200, 500, 1000, 2000]

    # run dev optimization
    best_config_bm25f = evaluate(BM25F, dev_transcripts, lambdas, True)
    print(("scoring_function", "lambda", "map", "mrr", "precision_at_" + str(RETURN_RANGE), "map_probs",
           "mrr_probs", "precision_at_" + str(RETURN_RANGE) + "_probs"))
    print(best_config_bm25f)
    best_config_tfidf = evaluate(TF_IDF, dev_transcripts, lambdas, True)
    print(("scoring_function", "lambda", "map", "mrr", "precision_at_" + str(RETURN_RANGE), "map_probs",
           "mrr_probs", "precision_at_" + str(RETURN_RANGE) + "_probs"))
    print(best_config_tfidf)

    # run tests to get real performance values
    print("-----------------------TEST-----------------------")

    # load test data
    test_transcripts = convert_to_json_list(path_test_set)

    # run tests for performance measurement
    _, real, predicted, all_classes = do_search(best_config_tfidf[0], test_transcripts, best_config_tfidf[1], True)
    # plot_confusion_matrix(real, predicted, all_classes, "TF_IDF")

    _, real, predicted, all_classes = do_search(best_config_bm25f[0], test_transcripts, best_config_bm25f[1], True)
    # plot_confusion_matrix(real, predicted, all_classes, "BM25")
