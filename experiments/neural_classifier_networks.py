"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: experiments
Date: 26.04.2019

"""
import argparse

import torch.nn as nn

from models.bert_classifier import BERTClassifier
from models.bert_with_conversation_context import BERTWithConversationContext
from utility.misc import get_device
from utility.model_parameter import Configuration
from utility.testing import evaluate_model
from utility.training import train_model

# parse command line args
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model", help="Changes the model use between CCN/LSTM/CE",
                        default="rnn", dest="model", choices=['siamese', 'bert', 'bert_context'])
arg_parser.add_argument("--skip_training", help="Skips the training and runs test process only", action="store_true",
                        default=False, dest="skip")
arg_parser.add_argument("--patience", help="Skips the training and runs test process only", action="store_true",
                        default=10, dest="patience")
arg_parser.add_argument("--bert_style", help="Skips the training and runs test process only", action="store_true",
                        default=False, dest="bert_style")
args = arg_parser.parse_args()

# load available device (cpu or cuda)
device, _ = get_device()

# configure if bert style preprocessing should be used (BPE)
bert_preprocessing = args.bert_style

# define model and load configuration
if args.model == "bert_context":
    model_class = BERTWithConversationContext
    config = Configuration("../configs.ini", "CLS_BERT_CONTEXT")
elif args.model == "bert":
    model_class = BERTClassifier
    config = Configuration("../configs.ini", "CLS_BERT")
elif args.model == 'history':
    model_class = BERTWithConversationContext
    config = Configuration("../configs.ini", "CONTEXT")

criterion = nn.CrossEntropyLoss()

# run training
model, data_fields, criterion = train_model(model_class, criterion, device, config, skip=args.skip,
                                            patience=args.patience,
                                            bert_preprocessing=bert_preprocessing)

# run test
evaluate_model(model, criterion, device, config, data_fields, shuffle=False)
