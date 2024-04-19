import pickle
import numpy as np
import preprocess as pre
import entropy_feature as enft
from sklearn.preprocessing import StandardScaler

X_path = "X.txt"
sample_rate = 512

# eeg_data: a numpy array with shape (5*sample_rate,)
def get_realtime_classification(eeg, sample_rate):
    nyquist_freq = 0.5 * sample_rate

    eeg_data = pre.mean_normalization(eeg)
    notch_filtered_data = pre.Notch_Filter(eeg_data, sample_rate)
    filtered_data = pre.BandFilter(notch_filtered_data, sample_rate)

    # get entropy feature
    AE = enft.AE(filtered_data)
    SE = enft.SE(filtered_data)
    FE = enft.FE(filtered_data)
    PE = enft.PE(filtered_data)

    # get frequency feature
    sig_fft, freq_fft = pre.FFT_ham(filtered_data, sample_rate)
    sig_power = np.abs(sig_fft) ** 2
    useful_sig_power = sig_power[:int(nyquist_freq * 5)]
    delta = np.mean(useful_sig_power[int(0.5 * 5):int(4 * 5)])
    theta = np.mean(useful_sig_power[int(4 * 5):int(8 * 5)])
    alpha = np.mean(useful_sig_power[int(8 * 5):int(13 * 5)])
    beta = np.mean(useful_sig_power[int(13 * 5):int(30 * 5)])
    gamma = np.mean(useful_sig_power[int(30 * 5):int(48 * 5)])
    fi = np.mean(useful_sig_power[int(0.85 * 5):int(110 * 5)])

    X = np.array([AE, SE, FE, PE, delta, theta, alpha, beta, gamma, fi]).reshape(-1, 10)
    # X = np.loadtxt(X_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = pickle.load(open('model_RF.pkl', 'rb'))
    y_pred = model.predict(X_scaled)

    return y_pred
