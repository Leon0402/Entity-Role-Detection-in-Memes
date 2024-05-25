import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from . import config
from .test_eval import EvalTest

TARGET_NAMES = ['other', 'victim', 'villain', 'hero']


def calculate_accuracy(outputs, labels, type_acr='batch'):
    if type_acr == 'batch':
        labels = labels.view(outputs.shape[0], -1)
        labels = labels.cpu().detach().numpy().tolist()
        outputs = torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist()
    outputs = np.argmax(outputs, axis=1)
    return metrics.f1_score(labels, outputs, average='macro')


def get_prediction_scores(predicted, y_test):
    precision, recall, fscore, support = score(y_test, predicted)
    accuracy = accuracy_score(y_test, predicted)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': fscore
    }


def train_fn(data_loader,
             model,
             optimizer,
             device,
             scheduler,
             accumulation_steps=config.ACCUMULATION_STEPS):
    model.train()

    total_loss = 0
    total_accuracy = 0
    pred_list = []
    gold_list = []
    with tqdm(enumerate(data_loader), unit="batch",
              total=len(data_loader)) as tepoch:
        for batch_index, dataset in tepoch:
            tepoch.set_description(f"Epoch Started")
            input_ids = dataset['input_ids']
            attention_mask = dataset['attention_mask']
            token_type_ids = dataset['token_type_ids']
            sentiment_target = dataset['labels']
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            sentiment_target = sentiment_target.to(device, dtype=torch.long)

            optimizer.zero_grad()

            outputs = model(input_ids,
                            attention_mask,
                            token_type_ids,
                            labels=sentiment_target)

            logits = outputs.logits
            loss = outputs.loss
            loss.backward()

            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            train_accuracy = 100.0 * calculate_accuracy(
                logits, sentiment_target)
            tepoch.set_postfix(loss=loss.item(), accuracy=train_accuracy)

            if (batch_index + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            total_accuracy += train_accuracy

            # ------------------------------------ New Addition ------------------------------------

            preds = torch.argmax(logits,
                                 axis=1).detach().cpu().numpy().tolist()
            gold = sentiment_target.cpu().tolist()

            pred_list.extend(preds)
            gold_list.extend(gold)

    # print("-"*50)
    # print("Printing the sets for the pred list and gold list")
    # print(set(pred_list))
    # print(set(gold_list))
    # print("-"*50)
    train_results = get_prediction_scores(pred_list, gold_list)
    train_cr = classification_report(y_true=gold_list,
                                     y_pred=pred_list,
                                     output_dict=True,
                                     target_names=TARGET_NAMES)
    train_cm = confusion_matrix(y_true=gold_list, y_pred=pred_list)

    # ----------------------------------------- Train Results -----------------------------------------

    print(
        "\n\ntrain_acc: {}\ttrain_precision: {}\ttrain_recall: {}\ttrain_f1: {}"
        .format(train_results['accuracy'], train_results['precision'],
                train_results['recall'], train_results['f1-score']))

    print(
        "\ntrain_hero_precision: {}\ttrain_hero_recall: {}\ttrain_hero_f1: {}".
        format(train_cr['hero']['precision'], train_cr['hero']['recall'],
               train_cr['hero']['f1-score']))

    print(
        "\ntrain_villain_precision: {}\ttrain_villain_recall: {}\ttrain_villain_f1: {}"
        .format(train_cr['villain']['precision'],
                train_cr['villain']['recall'],
                train_cr['villain']['f1-score']))

    print(
        "\ntrain_victim_precision: {}\ttrain_victim_recall: {}\ttrain_victim_f1: {}"
        .format(train_cr['victim']['precision'], train_cr['victim']['recall'],
                train_cr['victim']['f1-score']))

    print(
        "\ntrain_other_precision: {}\ttrain_other_recall: {}\ttrain_other_f1: {}"
        .format(train_cr['other']['precision'], train_cr['other']['recall'],
                train_cr['other']['f1-score']))
    print("\ntrain confusion matrix: ", train_cm)

    return total_loss / len(data_loader), total_accuracy / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()

    final_outputs = []
    final_targets = []
    final_outputs_idx = []

    with torch.no_grad():
        for _, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_ids = dataset['input_ids']
            attention_mask = dataset['attention_mask']
            token_type_ids = dataset['token_type_ids']
            sentiment_target = dataset['labels']

            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            sentiment_target = sentiment_target.to(device, dtype=torch.float)

            outputs = model(input_ids,
                            attention_mask,
                            token_type_ids,
                            labels=sentiment_target)

            logits = outputs.logits

            final_targets.extend(
                sentiment_target.cpu().detach().numpy().tolist())
            final_outputs.extend(
                torch.sigmoid(logits).cpu().detach().numpy().tolist())
            final_outputs_idx.extend(
                torch.argmax(logits, axis=1).detach().cpu().numpy().tolist())

    # print("-"*50)
    # print("Printing final_outputs")
    # print(final_outputs)
    # print("-"*50)
    # print("Printing final_targets")
    # print(final_targets)
    # print("-"*50)
    # print("Printing the argmax for the logits")
    # print(check_out)
    # print("-"*50)

    val_results = get_prediction_scores(final_outputs_idx, final_targets)
    val_cr = classification_report(y_true=final_targets,
                                   y_pred=final_outputs_idx,
                                   output_dict=True,
                                   target_names=TARGET_NAMES)
    val_cm = confusion_matrix(y_true=final_targets, y_pred=final_outputs_idx)

    # ----------------------------------------- Val Results -----------------------------------------

    print("\n\nval_acc: {}\tval_precision: {}\tval_recall: {}\tval_f1: {}".
          format(val_results['accuracy'], val_results['precision'],
                 val_results['recall'], val_results['f1-score']))

    print("\nval_hero_precision: {}\tval_hero_recall: {}\tval_hero_f1: {}".
          format(val_cr['hero']['precision'], val_cr['hero']['recall'],
                 val_cr['hero']['f1-score']))

    print(
        "\nval_villain_precision: {}\tval_villain_recall: {}\tval_villain_f1: {}"
        .format(val_cr['villain']['precision'], val_cr['villain']['recall'],
                val_cr['villain']['f1-score']))

    print(
        "\nval_victim_precision: {}\tval_victim_recall: {}\tval_victim_f1: {}".
        format(val_cr['victim']['precision'], val_cr['victim']['recall'],
               val_cr['victim']['f1-score']))

    print("\nval_other_precision: {}\tval_other_recall: {}\tval_other_f1: {}".
          format(val_cr['other']['precision'], val_cr['other']['recall'],
                 val_cr['other']['f1-score']))
    print("\nvalidation confusion matrix: ", val_cm)

    return final_outputs, final_targets
