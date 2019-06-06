import scipy.io.wavfile as wav
import numpy as np
from feature_extractor.webrtc_vad_filter import WebrtcVADFilter
from feature_extractor.python_speech_features import delta, rasta
# from feature_extractor.mfcc_feature_extractor import MfccFeatureExtractor
from feature_extractor.fir_filter import FIRFilter
import librosa


class FeatureExtractor:

    BASE_MODE = "mfcc-base"
    DELTA_MODE = "mfcc-delta"
    DELTA_DELTA_MODE = "mfcc-delta-delta"
    MODES = [BASE_MODE, DELTA_MODE, DELTA_DELTA_MODE]

    def __init__(self, data_files, extract_mode, vad_modes='none'):
        """
        data_files:
            A list of audio files
        """
        assert extract_mode in self.MODES

        self.feature_vectors = np.array([])
        self.extract_mode = extract_mode
        self.vad_filter = None

        if vad_modes == 'webrtc':
            self.vad_filter = WebrtcVADFilter(mode=3)

        self.win_length = 0.03

        # use webrtc VAD for filtering:
        for file in data_files:
            rate, signal = wav.read(file)
            original_len = len(signal)
            try:
                if self.vad_filter:
                    signal = self.vad_filter.filter_vad(signal, rate, 0.03)
                    if (len(signal)) == 0:
                        # error: file has no sound
                        continue
                        # os.unlink(file)
                print("File %s VAD filter result: %d/%d" % (file, len(signal), original_len))

                self.extract(signal, rate)
            except librosa.util.exceptions.ParameterError:
                pass
            print('Extracted file: {}'.format(file))

        # use FIR Filter for filtering:
        # for file in data_files:
        #     rate, signal = wav.read(file)
        #     filtered_signal = FIRFilter(signal, rate).f_filter()
        #     original_len = len(filtered_signal)
        #     try:
        #         if self.vad_filter:
        #             filtered_signal = self.vad_filter.filter_vad(filtered_signal, rate, 0.03)
        #             if (len(filtered_signal)) == 0:
        #                 # error: file has no sound
        #                 continue
        #         print("File %s VAD filter result: %d/%d" % (file, len(filtered_signal), original_len))
        #
        #         self.extract(filtered_signal, rate)
        #     except librosa.util.exceptions.ParameterError:
        #         pass
        #     print('Extracted file: {}'.format(file))

    def extract(self, signal, rate):
        rasta_feat = np.transpose(rasta.rastaplp(signal, fs=rate, modelorder=12))
        mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
        plp_feat = np.transpose(rasta.melfcc(signal, fs=rate, fbtype='bark'))

        if self.extract_mode == self.DELTA_MODE:
            mfcc_feat = np.concatenate((mfcc_feat, delta(mfcc_feat, 2)), axis=1)
        elif self.extract_mode == self.DELTA_DELTA_MODE:
            delta_feat = delta(mfcc_feat, 2)
            mfcc_feat = np.concatenate((mfcc_feat, delta_feat, delta(delta_feat, 2)), axis=1)

        feature_vector = np.concatenate((rasta_feat, mfcc_feat, plp_feat), axis=1)  # mfcc + rasta + plp
        # feature_vector = np.concatenate((mfcc_feat, plp_feat), axis=1)            # mfcc + plp
        # feature_vector = rasta_feat                                               # rasta
        # feature_vector = plp_feat                                                 # plp
        # feature_vector = mfcc_feat                                                # mfcc

        if self.feature_vectors.size == 0:
            self.feature_vectors = np.copy(feature_vector)
        else:
            self.feature_vectors = np.concatenate((self.feature_vectors, feature_vector))

        # return feature_vector

    def get_data(self):

        return self.feature_vectors


