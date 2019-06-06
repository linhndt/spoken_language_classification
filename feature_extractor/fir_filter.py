from scipy.signal import kaiserord, lfilter, firwin, freqz


class FIRFilter:

    def __init__(self, signal, rate):
        self.signal = signal
        self.rate = rate

    def f_filter(self, ripple=60, cutoff=5):
        nyq_rate = self.rate / 2
        width = 5.0 / nyq_rate
        N, beta = kaiserord(ripple, width)
        f_filter = firwin(N, cutoff/nyq_rate, width, window=('kaiser', beta))
        filtered_signal = lfilter(f_filter, 1, self.signal)

        return filtered_signal

