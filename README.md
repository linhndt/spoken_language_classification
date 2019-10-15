# SPOKEN LANGUAGE RECOGNITION

Tiny project for predicting the label of audio files according to the language of the voice inside.

The project is builded on the dataset given by the competition by Top Coder,
available at: https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16498&pm=13845

The project used multiple ML-based models for prediction: Random Forest, SVM, GMM with the input features extracted: 
MFCC, RASTA, PLP

## 0. Dependencies

1. Webrtc VAD:

VAD Library (use in MFCC).

[Github link](https://github.com/wiseman/py-webrtcvad)

```sh
cd <ANACONDA_BIN_FOLDER>
./conda install webrtcvad
```

## 1. Feature extractor

### MFCC

**Reference**:

[MFCC tutorial](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)

**Procedure**:

1. Frame the signal into short frames.
2. For each frame calculate the periodogram estimate of the power spectrum.
3. Apply the mel filterbank to the power spectra, sum the energy in each filter.
4. Take the logarithm of all filterbank energies.
5. Take the DCT of the log filterbank energies. (*result: cepstral coefficiens*)
6. Keep DCT coefficients 2-13, discard the rest.

**Explanation**:

1. Audio signal in short frame is considered stationary.
2. Calculate the power spectrum of each frame.
3. Mel filter bank allow us to sum up the energy in various frequency regions. Because the higher frequency, the harder for human ear to discern the difference between two closely spaced frequencies => the filters become larger at higher frequencies 
4. Because human ear perceive loudness not on linear scale, but on a scale similar to logarith scale (to double the perceived volume, we need to put 8 times energy)
5. Because the filter banks are overlapped => the filter bank energies are correlated. DCT allows to decorrelate the energies.
6. The high coefficients represent noise, so they will be discarded.

## 2. Usage

**Data** **Organization**:

Data folder contains:
* Training data folder: contains 'wav' files with the label correspondingly stated in 'trainingset.csv'
* Predicting data folder: contains 'wav' files for prediction.

**Predict** **audio** **label**:
1. Convert 'mp3' file format into 'wav' file format
2. Load data from *Training data folder* and *Predicting data folder*
3. Extract Features of those above files
4. Train model with training audio files
5. Predict labels of predicting audio files

**Testing** **procedure**:
1. Load data from *Training data folder*, the audio files in this folder will be divided into train, test dict
2. Extract Features of those files 
3. Train model with train dict
4. Evaluate results on test dict

