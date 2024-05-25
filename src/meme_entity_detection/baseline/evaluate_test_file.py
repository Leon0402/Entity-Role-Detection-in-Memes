from . import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from .test_eval import EvalTest
from .process_data import all_dataset

import transformers
import argparse
import re

transformers.logging.set_verbosity_error()

from .process_data import all_dataset, get_gold_label_test

parser = argparse.ArgumentParser()

parser.add_argument('-m',
                    '--model',
                    nargs='*',
                    default='deberta_small',
                    help='The name of the mdoel to be evaluated.')
parser.add_argument('-f',
                    '--face',
                    nargs='*',
                    default='1',
                    help='Whether or to use the face data.')

args = parser.parse_args()
model = args.model[0]
face = int(args.face[0])

if model == 'deberta_large':
    MODEL_NAME = "microsoft/deberta-v3-large"
    if face:
        MODEL_STORING_PATH = 'best_model_deberta_large_faces'
    else:
        MODEL_STORING_PATH = 'model_files/best_model_deberta_large'
else:
    MODEL_NAME = "microsoft/deberta-v3-small"
    if face:
        MODEL_STORING_PATH = 'best_model_deberta_small_faces'
    else:
        MODEL_STORING_PATH = 'model_files/best_model_deberta_small'

data_test = get_gold_label_test()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = torch.device(config.DEVICE)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                           num_labels=4)
model.load_state_dict(torch.load(MODEL_STORING_PATH))
model.to(device)
model.eval()

evals = EvalTest(tokenizer, model)
evals.get_test_eval(
    data_test,
    f"test_output_file_{re.sub('[^a-zA-Z0-9]', '_',MODEL_NAME)}.csv")
