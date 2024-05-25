# nohup python3 -u training_script.py -model deberta_large -face 1 > /home/temp/LCS2-Hero-Villain-Victim/out-files/baseline-constraint/logically-deberta-large-face.out &

import torch
from transformers import get_linear_schedule_with_warmup, AdamW, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from . import config
from .dataset import AspectRole
from .process_data import all_dataset, get_gold_label_test
from .engine import train_fn, eval_fn, calculate_accuracy
from .test_eval import EvalTest
import pandas as pd
import transformers
import numpy as np
import re
import argparse

transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser()

parser.add_argument('-model',
                    '--model',
                    nargs='*',
                    default='deberta_small',
                    help='The name of the mdoel to be evaluated.')
parser.add_argument('-face',
                    '--face',
                    nargs='*',
                    default='1',
                    help='Whether or to use the face data.')
parser.add_argument('-device',
                    '--device',
                    nargs='*',
                    default='gpu',
                    help='Whether or to use the face data.')

args = parser.parse_args()
model = args.model[0]
face = int(args.face[0])
device = args.device[0]
FACE_DATA = False

if model == 'deberta_large':
    MODEL_NAME = "microsoft/deberta-v3-large"
    if face:
        MODEL_STORING_PATH = 'model_files/best_model_deberta_large_faces'
        FACE_DATA = True
    else:
        MODEL_STORING_PATH = 'model_files/best_model_deberta_large'
else:
    MODEL_NAME = "microsoft/deberta-v3-small"
    if face:
        MODEL_STORING_PATH = 'model_files/best_model_deberta_small_faces'
        FACE_DATA = True
    else:
        MODEL_STORING_PATH = 'model_files/best_model_deberta_small'

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Using GPU: ", DEVICE)
else:
    DEVICE = torch.device("cpu")
    print("Using CPU: ", DEVICE)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(4)

data_train, data_val = all_dataset(FACE_DATA)
data_test = get_gold_label_test(FACE_DATA)

label2id = {'hero': 3, 'villain': 2, 'victim': 1, 'other': 0}

id2label = {id: tag for tag, id in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = AspectRole(tokenizer, data_train, label2id)
val_dataset = AspectRole(tokenizer, data_val, label2id)

train_loader = DataLoader(train_dataset,
                          batch_size=config.TRAIN_BATCH_SIZE,
                          shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=config.VALID_BATCH_SIZE)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label2id))
model.to(DEVICE)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        2e-5
    },
    {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0
    },
]

num_training_steps = int(
    len(data_train) / config.TRAIN_BATCH_SIZE * config.MAX_WHOLE_MODEL_EPOCHS)

optimizer = AdamW(optimizer_parameters, lr=config.LR)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

early_stopping_counter = 0
best_accuracy = 0

# for epochs in range(config.MAX_WHOLE_MODEL_EPOCHS):
#     print("Epoch :", epochs)
#     loss, train_accuracy = train_fn(train_loader, model, optimizer, DEVICE,
#                                     scheduler)
#     print(f"Total Epoch Train Accuracy : {train_accuracy} with loss : {loss}")
#     predicted, labels = eval_fn(val_loader, model, DEVICE)
#     val_accuracy = calculate_accuracy(predicted, labels, 'eval')
#     print(f"Total Epoch Eval Accuracy : {val_accuracy}")
#     if val_accuracy > best_accuracy:
#         early_stopping_counter = 0
#         best_accuracy = val_accuracy
#         torch.save(model.state_dict(), MODEL_STORING_PATH)
#     else:
#         early_stopping_counter += 1
#         if early_stopping_counter > config.EARLY_STOPPING_PATIENCE_WHOLE_MODEL:
#             break

model.load_state_dict(torch.load(MODEL_STORING_PATH))

print('Final Evaluation Test Data')
evals = EvalTest(tokenizer, model)
evals.get_test_eval(
    data_test,
    f"test_output_file_{re.sub('[^a-zA-Z0-9]', '_', MODEL_NAME)}.csv")
