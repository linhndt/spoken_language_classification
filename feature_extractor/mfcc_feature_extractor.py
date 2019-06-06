from feature_extractor.python_speech_features import mfcc, delta
from feature_extractor.webrtc_vad_filter import WebrtcVADFilter
import numpy as np
import scipy.io.wavfile as wav


class MfccFeatureExtractor:
    BASE_MODE = "mfcc-base"
    DELTA_MODE = "mfcc-delta"
    DELTA_DELTA_MODE = "mfcc-delta-delta"
    MODES = [BASE_MODE, DELTA_MODE, DELTA_DELTA_MODE]

    def __init__(self, data_files, extract_mode=BASE_MODE, vad_mode="webrtc"):
        '''

        :param data_files:
        :param extract_mode:
        :param vad: 'none', 'webrtc', or 'energy-base'
            use Voice Activity Detection algorithm
        '''
        assert extract_mode in self.MODES

        self.feature_vectors = np.array([])
        self.extract_mode = extract_mode
        self.vad_filter = None

        if vad_mode == "webrtc":
            self.vad_filter = WebrtcVADFilter(mode=3)

        self.win_length = 0.03
        for file in data_files:
            rate, signal = wav.read(file)
            original_len = len(signal)
            if self.vad_filter:
                signal = self.vad_filter.filter_vad(signal, rate, self.win_length)
                if (len(signal)) == 0:
                    # error: file has no sound
                    continue
            print("File %s VAD filtered result: %d/%d" % (file, len(signal), original_len))
            self.extract(rate, signal)

    def extract(self, rate, signal):
        # Because Webrtc VAD only support winlen 10 / 20 / 30 ms
        mfcc_feat = mfcc(signal, rate, winlen=self.win_length)

        if self.extract_mode == self.DELTA_MODE:
            mfcc_feat = np.concatenate((mfcc_feat, delta(mfcc_feat, 2)), axis=1)
        elif self.extract_mode == self.DELTA_DELTA_MODE:
            delta_feat = delta(mfcc_feat, 2)
            mfcc_feat = np.concatenate((mfcc_feat, delta_feat, delta(delta_feat, 2)), axis=1)

        if self.feature_vectors.size == 0:
            self.feature_vectors = np.copy(mfcc_feat)
        else:
            self.feature_vectors = np.concatenate((self.feature_vectors, mfcc_feat))

    def get_data(self):
        return self.feature_vectors
