from feature_extractor.mfcc_feature_extractor import MfccFeatureExtractor
from speaker_model.gmm_model import GaussianMixtureModel


class GmmSpeakerEnroller():

    def __init__(self, extract_mode):
        self.extract_mode = extract_mode
        self.speaker_models = dict()
        print("Initial GMM Speaker Enroller, extract method %s", extract_mode)

    def train(self, data_dict):
        print("Start training")
        self.speaker_models.clear()

        for speaker_label, files in data_dict.items():
            mfcc_extractor = MfccFeatureExtractor(files, self.extract_mode)
            speaker_model = GaussianMixtureModel(num_of_components=30)
            speaker_model.fit(mfcc_extractor.get_data())

            self.speaker_models[speaker_label] = speaker_model
            print("Finish training for speaker %s. Data shape %s",
                         speaker_label, str(mfcc_extractor.get_data().shape))

        print("Finish training")

    def get_label(self, file):
        data_points = MfccFeatureExtractor([file], self.extract_mode).get_data()

        result = dict()
        for label, model in self.speaker_models.items():
            result[label] = model.log_proba(data_points)

        return max(result, key=lambda key: result[key])
