"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: training.py
Date: 16.07.2019

"""
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer
from pytorch_transformers.optimization import AdamW
from torchtext.data import TabularDataset, Field, BucketIterator
from tqdm import tqdm

from preprocessing.generate_data_sets import NEGATIVE_SAMPLE_SIZE
from utility.model_logging import save_snapshot, load_snapshot, TrainingProtocol
from utility.model_parameter import ModelParameter, Configuration
from utility.testing import evaluate_epoch, get_loss

# set seed manually for repeatability
torch.manual_seed(1234)


def hinge_loss(correct_answer, incorrect_answer, margin):
    """
    Loss by calculating the correct/incorrect score difference in relation to given margin
    :param margin:
    :param correct_answer:
    :param incorrect_answer:
    :return:
    """
    loss_sum = torch.sum(margin + incorrect_answer - correct_answer)
    zero_vector = torch.Tensor([[0.0]]).to(loss_sum.device)
    return max(zero_vector, loss_sum)


def get_optimizer(model: nn.Module, name, **kwargs):
    """
    initializes the optimizer
    :param model:
    :param name:
    :param kwargs:
    :return:
    """
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif name == "Adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif name == "BertAdam":
        return AdamW(model.parameters(), **kwargs)


def load_data(training_file_path: str, dev_file_path: str, batch_size: int, max_length: int, device,
              is_bert_model=True):
    """
    loads training data from csv files
    :param batch_size:
    :param max_length:
    :param device:
    :param is_bert_model:
    :param training_file_path:
    :param dev_file_path:
    :return:
    """

    bert_tokenizer = BertTokenizer.from_pretrained(
        "bert-base-german-cased",
        do_lower_case=False)

    def tokenize(sent):
        return bert_tokenizer.tokenize(sent)

    def tokenize_sentence(sentence: str):
        if is_bert_model:
            sentence = "[CLS] " + sentence + " [SEP]"
        tokens = tokenize(sentence)
        return list(tokens)

    # set text fields
    TEXT_FIELD = Field(tokenize=tokenize_sentence, sequential=True, lower=False, fix_length=max_length,
                       pad_token="[PAD]",
                       batch_first=not is_bert_model, use_vocab=not is_bert_model)

    def numericalize(seq_lists, device=device):
        ids = []
        for seq in seq_lists:
            ids.append(bert_tokenizer.convert_tokens_to_ids(seq))
        return torch.LongTensor(ids).to(device)

    TEXT_FIELD.numericalize = numericalize

    # define static fields
    CLASS_FIELD = Field(sequential=False)
    POSITION_FIELD = Field(sequential=False, use_vocab=False)

    data_fields = [("sample_class", CLASS_FIELD), ("sample", TEXT_FIELD), ("position", POSITION_FIELD),
                   ("previous_classes", CLASS_FIELD), ('previous_sample', TEXT_FIELD),
                   ("sample_pos", TEXT_FIELD)] + [
                      ("sample_neg_" + str(i), TEXT_FIELD) for i in range(NEGATIVE_SAMPLE_SIZE)]

    # load and prepare training dataset
    train_data = TabularDataset(path=training_file_path,
                                format='tsv',
                                fields=data_fields, skip_header=True)

    dev_data = TabularDataset(path=dev_file_path,
                              format='tsv',
                              fields=data_fields, skip_header=True)

    train_iter = BucketIterator(train_data,
                                batch_size=batch_size,
                                device=device,
                                sort=True,
                                sort_key=lambda x: x.sample,
                                sort_within_batch=True,
                                shuffle=True
                                )

    dev_iter = BucketIterator(dev_data,
                              batch_size=batch_size,
                              device=device,
                              sort=True,
                              sort_key=lambda x: x.sample,
                              sort_within_batch=True,
                              shuffle=True
                              )

    # build class vocab
    CLASS_FIELD.build_vocab(train_data)

    if not is_bert_model:
        TEXT_FIELD.build_vocab(train_data)

    return train_iter, dev_iter, CLASS_FIELD, TEXT_FIELD, data_fields


class EarlyStopping:
    def __init__(self, initial_scorce, patience):
        """
        Stops the training once there is no significant improvement over the last epochs
        (patience = number of epochs to wait for improvement
        :param initial_scorce:
        :param patience:
        """
        self.score = initial_scorce
        self.patience = patience
        self.epoch_counter = 0

    def update(self, score):
        if score > self.score:
            self.epoch_counter = 0
            self.score = score

        else:
            self.epoch_counter += 1
            if self.epoch_counter > self.patience:
                return True
        return False


def train_model(model_class, criterion, device, config: Configuration, skip=False, patience=5, task="classification",
                bert_preprocessing=True, margin=0.2, log_train_loss=False):
    """
    trains a model using the provide configuration
    :param skip:
    :param model:
    :param config:
    :return:
    """
    # load data
    TRAINING_DATA_FILE = config.get_string(ModelParameter.TRAINING_FILE)
    DEV_DATA_FILE = config.get_string(ModelParameter.DEV_FILE)
    BATCH_SIZE = config.get_int(ModelParameter.BATCH_SIZE)
    MAX_LENGTH = config.get_int(ModelParameter.MAX_LENGTH)

    train_iterator, dev_iterator, class_field, text_field, data_fields = load_data(TRAINING_DATA_FILE, DEV_DATA_FILE,
                                                                                   BATCH_SIZE,
                                                                                   MAX_LENGTH, device=device,
                                                                                   is_bert_model=bert_preprocessing)

    # initialize optimizer and model
    EPOCHS = config.get_int(ModelParameter.EPOCHS)
    model = model_class(config, len(class_field.vocab.freqs) + 1, class_field=class_field, device=device).to(device)
    LR = config.get_float(ModelParameter.LEARNING_RATE)
    WEIGHT_DECAY = config.get_float(ModelParameter.WEIGHT_DECAY)
    OPTIMIZER_NAME = config.get_string(ModelParameter.OPTIMIZER)
    OPTIMIZER = get_optimizer(model, OPTIMIZER_NAME, **{"lr": LR, "weight_decay": WEIGHT_DECAY})

    # parameter to skip training steps
    if not skip:
        # prepare logging
        protocol = TrainingProtocol("../protocols/", model_class.__name__, ["map", "acc"])

        # run trainings epochs and keep best performance score
        print("Starting training...")
        max_acc_score = .0
        early_stopping = EarlyStopping(max_acc_score, patience=patience)
        for epoch in range(1, EPOCHS + 1):
            print("\n", "Epoch %i of %i" % (epoch, EPOCHS), "\n")

            model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_iterator):

                loss, _ = get_loss(model, batch, task, criterion, device, margin=margin)
                epoch_loss += loss.item()

                if loss.item() > 0:
                    OPTIMIZER.zero_grad()
                    loss.backward()
                    OPTIMIZER.step()

                del loss

            print("Epoch Loss: %f" % epoch_loss)
            acc, map, dev_loss = evaluate_epoch(model, dev_iterator, criterion, batch_size=BATCH_SIZE, device=device,
                                                task=task)

            if acc > max_acc_score:
                max_acc_score = acc
                save_snapshot(model, OPTIMIZER, epoch, epoch_loss, max_acc_score, "../snapshots")

            if log_train_loss:
                train_acc, _, epoch_loss = evaluate_epoch(model, dev_iterator, criterion, batch_size=BATCH_SIZE,
                                                          device=device,
                                                          task=task)
            protocol.log_epoch(epoch, epoch_loss, dev_loss, map, acc)

            del epoch_loss

            if early_stopping.update(max_acc_score):
                print("Stopping training...")
                break

    # loads the best performing model parameters and sets them in the model object
    load_snapshot(model, directory="../snapshots")

    return model, data_fields, criterion
