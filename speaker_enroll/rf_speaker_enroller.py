
from sklearn.ensemble import RandomForestClassifier

from feature_extractor.features_extractor import FeatureExtractor
import numpy as np
from collections import Counter
import os
import pickle


class RandomForestSpeakerEnroller():

    def __init__(self, extract_mode):
        self.extract_mode = extract_mode
        self.clf = RandomForestClassifier(n_estimators=10)
        # self.pca = PCA(n_components=20)
        # self.params = {'n_estimators': [n for n in [10, 200, 500, 1000]]}
        # self.grid_search = GridSearchCV(self.clf, self.params, cv=5, n_jobs=-1)

    def train(self, data_dict):

        pkl_file = '/home/ndthlinh/PycharmProjects/spoken_language_classification/rf_classifier.pkl'

        if os.path.isfile(pkl_file):

            print('Using pre-trained model ....')

        else:

            x = None  # data points
            y = list()  # data labels

            for label, training_files in data_dict.items():
                file_data = FeatureExtractor(training_files, self.extract_mode).get_data()
                # file_data = FeatureExtractor(training_files).get_data()
                x = np.copy(file_data) if x is None else np.concatenate((x, file_data))

                y += [label] * len(file_data)

            # self.grid_search.fit(x, y)
            # self.pca.fit(x)
            # x = self.pca.transform(x)
            self.clf.fit(x, y)

            # save model:
            pickle.dump(self.clf, open(pkl_file, 'wb'))

            print("Finish training random forest")

    def get_label(self, file):

        data_points = FeatureExtractor([file], self.extract_mode).get_data()
        # data_points = self.pca.transform(data_points)
        # data_points = FeatureExtractor([file]).get_data()
        # predict_result = self.grid_search.predict(data_points).tolist()

        # Load pre-trained model:
        pkl_file = '/home/ndthlinh/PycharmProjects/spoken_language_classification/rf_classifier.pkl'

        if os.path.isfile(pkl_file):

            loaded_model = pickle.load(open(pkl_file, 'rb'))

            if data_points.size == 0:
                predict_result = 'noise'
            else:
                predict_result = loaded_model.predict(data_points).tolist()

        else:

            predict_result = self.clf.predict(data_points).tolist()

        most_common_label, num_most_common_label = Counter(predict_result).most_common(1)[0]
            # print(self.grid_search.best_params_)

        return most_common_label
        # return predict_result

