# -*- coding:utf-8 -*-
from process_email import ProcessEmail
import os
import numpy as np

def EmailFeatures(word_indices):
    n = 1899
    x = [0]*n
    for i in word_indices:
        x[i] = 1
    return x


def readfile(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        file_contents = f.readlines()
    f.close()
    return file_contents


def get_train_test_dataset(path, train_test_ratio):
    all_email_features = np.zeros((1, 1899), dtype=int)
    all_email_labels = [0]

    for i in os.listdir(path):
        path_1 = path + '\\' + i
        # print(path_1)

        # get features and labels
        for j in os.listdir(path_1):
            # get labels
            if "hard" in path_1:
                all_email_labels.append(0)
            elif "easy" in path_1:
                all_email_labels.append(0)
            else:
                all_email_labels.append(1)
            # get features
            path_2 = path_1 + '\\' + j
            print("Reading " + path_2)
            file_contents = readfile(path_2)
            word_indices = ProcessEmail(file_contents)
            features = np.array(EmailFeatures(word_indices)).reshape(1, 1899)
            all_email_features = np.concatenate((all_email_features, features), axis=0)
    all_email_labels = np.array(all_email_labels).reshape(len(all_email_labels), 1)

    x_train = np.zeros((1, 1899), dtype=int)
    x_test = np.zeros((1, 1899), dtype=int)
    y_train = np.zeros((1, 1), dtype=int)
    y_test = np.zeros((1, 1), dtype=int)
    for i in range(len(all_email_features)):
        if i <= int(train_test_ratio * len(all_email_features)):
            x_train = np.concatenate((x_train, all_email_features[i].reshape(1, 1899)), axis=0)
        else:
            x_test = np.concatenate((x_test, all_email_features[i].reshape(1, 1899)), axis=0)

    for j in range(len(all_email_labels)):
        if j <= int(train_test_ratio * len(all_email_labels)):
            y_train = np.concatenate((y_train, all_email_labels[j].reshape(1, 1)), axis=0)
        else:
            y_test = np.concatenate((y_test, all_email_labels[j].reshape(1, 1)), axis=0)
    return x_train, y_train, x_test, y_test



