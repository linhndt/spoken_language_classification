import webrtcvad
from scipy.io import wavfile
import scipy
import numpy as np

class WebrtcVADFilter:
    """
    NOTE: WebRTC VAD only accepts 16 bit mono PCM audio, sampled at 8000, 16000, 32000, or 48000 Hz
    and window length (frame length) of 10, 20 or 30 ms
    """

    def __init__(self, mode=3):
        self.webrtc_vad = webrtcvad.Vad(mode)

    def filter_vad(self, signal, rate, winlen):
        assert winlen in [0.01, 0.02, 0.03]

        const = float(2 ** 15)
        audio_n = signal / const
        Z = self._audioSlice(audio_n, rate, winlen)
        num_of_vad = 0
        vad_signal = np.array([])
        for i, z in enumerate(Z):
            fr = np.int16(z * 32768).tobytes()
            if self.webrtc_vad.is_speech(fr, rate):
                vad_signal = np.append(vad_signal, z)


        return vad_signal

    def _audioSlice(self, x, rate, winlen):
        framesamp = int(winlen*rate)
        X = scipy.array([x[i:i+framesamp] for i in range(0, len(x)-framesamp, framesamp)])
        return X
