"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: experiments
Date: 26.04.2019

"""
import argparse

from models.siamese_neural_network import SiameseNeuralNetwork
from utility.misc import get_device
from utility.model_parameter import Configuration, ModelParameter
from utility.testing import evaluate_model
from utility.training import train_model, hinge_loss

# parse command line args
arg_parser: argparse.ArgumentParser = argparse.ArgumentParser()
arg_parser.add_argument("--model", help="Changes the model",
                        default="cnn", dest="model", choices=['bert', 'cnn'])
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
model_class = SiameseNeuralNetwork

if args.model == 'cnn':
    config = Configuration("../configs.ini", "RANK_CNN")
elif args.model == 'bert':
    config = Configuration("../configs.ini", "RANK_BERT")

criterion = hinge_loss
MARGIN = config.get_float(ModelParameter.MARGIN)

# run training
model, data_fields, _ = train_model(model_class, criterion, device, config, skip=args.skip, patience=args.patience,
                                    task="ranking", margin=MARGIN, bert_preprocessing=bert_preprocessing)

# run test
evaluate_model(model, criterion, device, config, data_fields, task="ranking", margin=MARGIN)
