from speaker_enroll import *
from datetime import datetime

from sklearn.metrics import confusion_matrix
from ultilities import data_loader


def get_enroller(enroller_type, extract_mode):
    """
    Load enroller types
    """

    assert enroller_type in [
        "gmm",      # Gaussian model
        "rf",       # Random forest
        "l_svm",    # Linear SVM
        "g_svm"     # Gaussian (RBF) SVM
    ]

    if enroller_type == "gmm":
        return GmmSpeakerEnroller(extract_mode)
    elif enroller_type == "rf":
        return RandomForestSpeakerEnroller(extract_mode)
    elif enroller_type == "l_svm":
        return SVMSpeakerEnroller(extract_mode, "linear")
    else:
        return SVMSpeakerEnroller(extract_mode, "rbf")


def test_enroller(corpus_folders, enroller, train_test_ratio=0.7):

    train_dict, test_dict = data_loader.train_test_split(corpus_folders, train_test_ratio)

    time_mark = datetime.now()

    enroller.train(train_dict)

    train_time = datetime.now() - time_mark
    print("Finish training in %s" % str(train_time))
    time_mark = datetime.now()

    # Test
    all_correct_predict = 0
    all_total_predict = 0

    y_true = list()
    y_pred = list()
    labels = list()

    for speaker_label, test_files in test_dict.items():
        speaker_correct_predict = 0
        speaker_total_predict = 0

        labels.append(speaker_label)

        for test_file in test_files:
            predict_label = enroller.get_label(test_file)

            if speaker_label == predict_label:

                speaker_correct_predict += 1

            y_true.append(speaker_label)
            y_pred.append(predict_label)

            speaker_total_predict += 1

            print(
                "Speaker %s predict as %s result %.2f%%. Correct: %d/%d" % (speaker_label, predict_label,
                                                                            speaker_correct_predict * 100 / speaker_total_predict,
                                                                            speaker_correct_predict,
                                                                            speaker_total_predict))

        all_correct_predict += speaker_correct_predict
        all_total_predict += speaker_total_predict

    test_time = datetime.now() - time_mark
    cf_matrix = confusion_matrix(y_true, y_pred)
    return all_correct_predict, all_total_predict, train_time, test_time, cf_matrix, labels


if __name__ == '__main__':

    data_path = 'data/training_dataset'

    start_time = datetime.now()
    test_times = 1

    extract_mode = FeatureExtractor.DELTA_DELTA_MODE
    enroller_type = "gmm"
    enroller = get_enroller(enroller_type, extract_mode)

    test_results = []

    for i in range(0, test_times):
        all_correct_predict, all_predict, train_time, test_time, cf_matrix, labels = test_enroller(data_path, enroller)
        accuracy = all_correct_predict * 100 / all_predict

        test_results.append((accuracy, cf_matrix))


    print(test_results)

    # end_time = datetime.now() - start_time
    print("Time running for .. model is : {}s".format(datetime.now() - start_time))
