from . import config

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from role_label_test_eval_batch import EvalTest

import transformers
# import argparse
import re

transformers.logging.set_verbosity_error()

# parser = argparse.ArgumentParser()

# parser.add_argument('-m', '--model', nargs='*', default='deberta_large', help='The name of the model to be evaluated.')

# args = parser.parse_args()
# model = args.model[0]

model = config.MODEL

# print(model)


def process_data():
    # Create and return a df for a total of N samples
    N = 1  # can be dynamically set as the total samples to be processed
    original = []  # Raw OCR from meme image
    text = []  # processed OCR text
    word = []  # entity label being probed
    image = []  # image name

    possible_OCRs = [
        "Donald Trump has been completely responsible for all the human index development in America.",
        "Donald Trump is completely responsible for all the human rights violation that America has ever seen.",
        "Donald Trump has been completely ignored during his presidency.",
        "Donald Trump was the previous president of United States of America.",
        "Is there anything that Donald Trump hasn't spoiled yet?"
    ]

    for i in range(N):
        # print(i)
        original_ocr = possible_OCRs[i]
        original.append(original_ocr)
        text.append(original_ocr.lower().replace('\n', ' '))
        word.append("Donald Trump")
        image.append("text_meme_{}.jpg".format(str(i)))

    df_test = pd.DataFrame()
    df_test['sentence'] = text
    df_test['original'] = original
    df_test['word'] = word
    df_test['image'] = image
    return df_test


print('------------------------------------------')
print(model)
if model == 'deberta_large':
    MODEL_NAME = "microsoft/deberta-v3-large"
    MODEL_STORING_PATH = 'model_files/best_model_deberta_large'
else:
    MODEL_NAME = "microsoft/deberta-v3-small"
    MODEL_STORING_PATH = 'model_files/best_model_deberta_small'

# -------------------------------------------------------------------------------------
# Define processing module (replaces the module that returns the test set dataframe)
# This can be reverse mapped to the pipeline feeding meme details: oce, entity, image name etc.
data_test = process_data()

# data_test = get_gold_label_test()

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
