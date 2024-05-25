import pandas as pd
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from sklearn.utils import resample


def all_dataset(face_data=False):
    train_covid = './data/HVVMemes/annotations/train.jsonl'
    # train_pol = 'path-to-uspol-train.jsonl'

    json_data_train = []

    with open(train_covid, 'r') as json_file:
        for json_str in list(json_file):
            json_data_train.append(json.loads(json_str))

    # with open(train_pol, 'r') as json_file:
    #     for json_str in list(json_file):
    #         json_data_train.append(json.loads(json_str))

    val_covid = './data/HVVMemes/annotations/dev.jsonl'
    # val_pol = 'path-to-uspol-val.jsonl'

    json_data_val = []

    with open(val_covid, 'r') as json_file:
        for json_str in list(json_file):
            json_data_val.append(json.loads(json_str))

    # with open(val_pol, 'r') as json_file:
    #     for json_str in list(json_file):
    #         json_data_val.append(json.loads(json_str))

    if face_data:
        train_face_dict, val_face_dict = load_faces_data()

    original = []
    text = []
    word = []
    role = []

    for vals in tqdm(json_data_train):
        sentence = vals['OCR'].lower().replace('\n', ' ')
        for keys in ['hero', 'villain', 'victim', 'other']:
            for word_val in vals[keys]:
                original.append(vals['OCR'])
                if face_data:
                    text.append(sentence + '\n' +
                                ' '.join(train_face_dict[vals['image']]))
                else:
                    text.append(sentence)
                word.append(word_val)
                role.append(keys)

    df_train = pd.DataFrame()

    df_train['sentence'] = text
    df_train['original'] = original
    df_train['word'] = word
    df_train['role'] = role

    original = []
    text = []
    word = []
    role = []

    for vals in tqdm(json_data_val):
        sentence = vals['OCR'].lower().replace('\n', ' ')
        for keys in ['hero', 'villain', 'victim', 'other']:
            for word_val in vals[keys]:
                original.append(vals['OCR'])
                if face_data:
                    text.append(sentence + '\n' +
                                ' '.join(val_face_dict[vals['image']]))
                else:
                    text.append(sentence)
                word.append(word_val)
                role.append(keys)

    df_val = pd.DataFrame()

    df_val['sentence'] = text
    df_val['original'] = original
    df_val['word'] = word
    df_val['role'] = role

    df_other = df_train[df_train.role == 'other']

    df_hero = df_train[df_train.role == 'hero']
    df_villian = df_train[df_train.role == 'villain']
    df_victim = df_train[df_train.role == 'victim']

    df_hero_upsampled = resample(
        df_hero,
        replace=True,  # sample with replacement
        n_samples=2000,  # to match majority class
        random_state=42)  # reproducible results

    df_other = pd.concat([df_other, df_hero])
    df_other = pd.concat([df_other, df_hero_upsampled])

    df_villian_upsampled = resample(
        df_villian,
        replace=True,  # sample with replacement
        n_samples=2000,  # to match majority class
        random_state=42)  # reproducible results

    df_other = pd.concat([df_other, df_villian])
    df_other = pd.concat([df_other, df_villian_upsampled])

    df_victim_upsampled = resample(
        df_victim,
        replace=True,  # sample with replacement
        n_samples=2000,  # to match majority class
        random_state=42)  # reproducible results

    df_train_final = pd.concat([df_other, df_victim])
    df_train_final = pd.concat([df_train_final, df_victim_upsampled])

    return df_train_final, df_val


def load_faces_data():
    train_face_dict = {}
    val_face_dict = {}

    covid_file_path = 'path-to-covid19-annotations/'
    uspol_file_path = 'path-to-uspolitics-annotations/'

    json_data_train = []

    with open(
            covid_file_path +
            'train_ocr_match-v2-subImages-imagenet22k_infos.jsonl',
            'r') as json_file:
        for json_str in list(json_file):
            json_data_train.append(json.loads(json_str))

    with open(
            uspol_file_path +
            'val_ocr_match-v2-subImages-imagenet22k_infos.jsonl',
            'r') as json_file:
        for json_str in list(json_file):
            json_data_train.append(json.loads(json_str))

    for vals in json_data_train:
        train_face_dict[vals['image']] = vals['faces']

    json_data_val = []

    with open(
            covid_file_path +
            'train_ocr_match-v2-subImages-imagenet22k_infos.jsonl',
            'r') as json_file:
        for json_str in list(json_file):
            json_data_val.append(json.loads(json_str))

    with open(
            uspol_file_path +
            'val_ocr_match-v2-subImages-imagenet22k_infos.jsonl',
            'r') as json_file:
        for json_str in list(json_file):
            json_data_val.append(json.loads(json_str))

    for vals in json_data_val:
        val_face_dict[vals['image']] = vals['faces']

    return train_face_dict, val_face_dict


def get_gold_label_test(faces_data=False):
    file_path = './data/HVVMemes/annotations/dev_test.jsonl'

    json_data_val = []

    with open(file_path, 'r') as json_file:
        for json_str in list(json_file):
            json_data_val.append(json.loads(json_str))

    if faces_data:
        test_face_dict = {}
        face_json_data = []

        with open(
                'path-to-unseen_test-faces-0.95-subImages-imagenet22k_infos-golds.jsonl',
                'r') as json_file:
            for json_str in list(json_file):
                face_json_data.append(json.loads(json_str))

        for vals in face_json_data:
            test_face_dict[vals['image']] = vals['faces']

    original = []
    text = []
    word = []
    image = []
    role = []

    for vals in tqdm(json_data_val):
        sentence = vals['OCR'].lower().replace('\n', ' ')
        for keys in ['hero', 'villain', 'victim', 'other']:
            for word_val in vals[keys]:
                original.append(vals['OCR'])
                if faces_data:
                    text.append(sentence + '\n' +
                                ' '.join(test_face_dict[vals['image']]))
                else:
                    text.append(sentence)
                word.append(word_val)
                role.append(keys)
                image.append(vals['image'])

    df_train = pd.DataFrame()

    df_train['sentence'] = text
    df_train['original'] = original
    df_train['word'] = word
    df_train['image'] = image
    df_train['role'] = role

    return df_train
