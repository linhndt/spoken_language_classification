import pandas as pd
from collections import defaultdict
import os


def read_csv_file(file):

    if os.path.isfile(file):
        data = pd.read_csv(file, header=None)

    else:
        raise ValueError('Retry entering your path')

    return data


def train_test_split(data_folder, train_test_ratio):
    """
        Return the dict of speakers train files (key: speaker label, value: list of files)
        and dict of test files (key: speaker label, value: list of files)

        Parameters
        --------------

        data_folders: list
            A list of corpus folders. Each corpus folder must be organized as a set of speaker folders

        train_test_ratio: float
            The ratio between training set and test set. Must be in range of (0; 1)

        Return
        --------------
        train_set:  dict
            A dictionary of training files. Key: speaker label, value: list of training files

        test_set: dict
            A dictionary of test files. Key: speaker label, value: list of test files

    """

    training_csv = 'trainingset.csv'

    data = read_csv_file(training_csv)

    file_name_list = []
    label_list = []

    for index, row in data.iterrows():

        file_name = data_folder + '/' + row[0][:-4] + '.wav'
        label = row[1]

        file_name_list.append(file_name)
        label_list.append(label)

    # Split into train, test dict:

    assert 1 > train_test_ratio > 0

    num_of_training_files = int(len(file_name_list) * train_test_ratio)

    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    train_file_list = file_name_list[: num_of_training_files]
    train_label_list = label_list[:num_of_training_files]

    test_file_list = file_name_list[num_of_training_files:]
    test_label_list = label_list[num_of_training_files:]

    for file_name, label in zip(train_file_list, train_label_list):
        train_dict[label].append(file_name)

    # for k, v in train_dict.items():
    #     print('{}, {}, {}'.format(k, len(v), v))

    for file_name, label in zip(test_file_list, test_label_list):
        test_dict[label].append(file_name)

    # for k, v in test_dict.items():
    #     print('{}, {}, {}'.format(k, len(v), v))

    return train_dict, test_dict


def load_training_data(training_data_folder):
    """
            Return the dict of speakers train files (key: speaker label, value: list of files)

            Parameters
            --------------

            training_data_folder: path
                Contain 'wav' file for training purpose

            Return
            --------------
            train_set:  dict
                A dictionary of training files. Key: speaker label, value: list of training files

        """
    training_csv = 'trainingset.csv'

    data = read_csv_file(training_csv)

    file_name_list = []
    label_list = []

    for index, row in data.iterrows():
        file_name = training_data_folder + '/' + row[0][:-4] + '.wav'
        label = row[1]

        file_name_list.append(file_name)
        label_list.append(label)

    train_dict = defaultdict(list)

    for file_name, label in zip(file_name_list, label_list):
        train_dict[label].append(file_name)

    # for k, v in train_dict.items():
    #     print('{}, {}, {}'.format(k, len(v), v))

    return train_dict


def load_predict_folder(predict_data_folder):

    predict_list = []

    for file in os.listdir(predict_data_folder):

        file_name = predict_data_folder + '/' + file
        predict_list.append(file_name)

    return predict_list