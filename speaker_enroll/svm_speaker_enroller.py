from sklearn import svm
from sklearn.preprocessing import StandardScaler
from feature_extractor.features_extractor import FeatureExtractor
from feature_extractor.mfcc_feature_extractor import MfccFeatureExtractor

import numpy as np

LINEAR = "linear"
RBF = "rbf"


class SVMSpeakerEnroller():
    """
    The SVM approach need to enroll all speakers data at once
    """

    def __init__(self, extract_mode=FeatureExtractor.DELTA_DELTA_MODE, svm_kernel=LINEAR):
        self.extract_mode = extract_mode
        self.kernel = svm_kernel
        self.scaler = StandardScaler()
        self.clf = svm.LinearSVC() if self.kernel == LINEAR else svm.SVC(kernel=self.kernel)

    def train(self, data_dict):
        x = None  # data points
        y = list()  # data labels
        for label, training_files in data_dict.items():
            file_data = FeatureExtractor(training_files, extract_mode=self.extract_mode).get_data()
            x = np.copy(file_data) if x is None else np.concatenate((x, file_data))

            y += [label] * len(file_data)

        print("Training data point:", x.shape)
        print("Start perform feature scaling")

        x = self.scaler.fit_transform(x)
        self.clf.fit(x, y)

    def get_label(self, file):
        data_points = FeatureExtractor([file], extract_mode=self.extract_mode).get_data()

        class_points = [0] * len(self.clf.classes_)

        scaled_data_points = self.scaler.transform(data_points)
        dec = self.clf.decision_function(scaled_data_points)

        for sub_dec in dec:
            print("Sub decision:", sub_dec)
            # predict_label_index = np.where(sub_dec == max(sub_dec))[0][0]
            predict_label_index = 1 if sub_dec > 0 else 0
            class_points[predict_label_index] += 1

        # get the class which most data points belong to
        return self.clf.classes_[class_points.index(max(class_points))]

