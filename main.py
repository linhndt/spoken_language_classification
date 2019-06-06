from speaker_enroll import *
from datetime import datetime
from feature_extractor.features_extractor import FeatureExtractor
from sklearn.metrics import confusion_matrix
from ultilities import data_loader


def get_enroller(enroller_type, extract_mode):
    """
    Load enroller for model

    Parameters
    ----------------
    enroller_type:
        Type of enroller which is the ML-based model for classification.
        List of enroller_types:
        "gmm",      # Gaussian model
        "rf",       # Random forest
        "l_svm",    # Linear SVM
        "g_svm"     # Gaussian (RBF) SVM

    extract_mode:
        Extract mode to extract MFCC features
        BASE_MODE = "mfcc-base"
        DELTA_MODE = "mfcc-delta"
        DELTA_DELTA_MODE = "mfcc-delta-delta"
        MODES = [BASE_MODE, DELTA_MODE, DELTA_DELTA_MODE]

    Return
    ----------------
    enroller_model:
         Return corresponding model with (enroller_type, extract_mode)
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


def main(training_folder='data/training_dataset', predict_folder='data/predicting_dataset', enroller_type='rf', extract_mode=FeatureExtractor.DELTA_DELTA_MODE):
    """
        Main function to run model for predicting the spoken_language_label of audio files.

        Parameters
        ----------------
        training_folder: path
            Folder contains training files in training phase.

        predict_folder: path
            Folder contains files for predicting labels

        enroller_type, extract_mode:
            Variables for get_enroller function to return the ML-based model for prediction

    """


    # Load training_data, predict_data:

    train_dict = data_loader.load_training_data(training_folder)
    predict_file_list = data_loader.load_predict_folder(predict_folder)
    enroller = get_enroller(enroller_type, extract_mode)

    time_mark = datetime.now()

    # Training:

    print("Start training")

    enroller.train(train_dict)

    train_time = datetime.now() - time_mark
    print("Finish training in %s" % str(train_time))

    # Predict:

    for file in predict_file_list:
        predict_label = enroller.get_label(file)
        print("The label of audio file {} is: {}".format(file, predict_label))


if __name__ == "__main__":
    main()
