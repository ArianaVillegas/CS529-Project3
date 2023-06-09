import os
import pandas as pd


def create_train_meta(data_dir, augmented=False):
    rows = []
    list_dir = os.listdir(data_dir)
    for idx, label in enumerate(list_dir):
        class_folder = os.path.join(data_dir, label)
        list_dir = os.listdir(class_folder)
        if augmented:
            for name in os.listdir(class_folder):
                if name == 'output':
                    list_dir += [os.path.join(name, sub) for sub in os.listdir(os.path.join(class_folder, name))]
        for name in list_dir:
            if name.endswith('.png'):
                rows.append([os.path.join(class_folder, name), label, idx])
    df = pd.DataFrame(rows, columns=['path', 'label', 'idx'])
    return df


def create_test_meta(data_dir):
    rows = []
    list_dir = os.listdir(data_dir)
    for idx, label in enumerate(list_dir):
        rows.append([os.path.join(data_dir, label), -1, label])
    df = pd.DataFrame(rows, columns=['path', 'label', 'idx'])
    return df