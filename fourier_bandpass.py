import numpy as np
import matplotlib.pyplot as plt

import scipy.fftpack as fftpack

# frequency bandpass filter using numpy's Fourier transforms

np.set_printoptions(threshold=np.nan)

# conversion to frequency and back (just load img with PIL)
# print(np.fft.irfft2(np.fft.rfft2(img)))
# plt.imshow(np.fft.irfft2(np.fft.rfft2(img)))

sample_rate = 128  # per second
time = np.arange(0, 10, 1 / sample_rate)  # from 0 to 10 seconds, with 1/rate mean-time between samples
signal = np.cos(4 * np.pi * time) + np.cos(6 * np.pi * time) + np.cos(8 * np.pi * time)


def spectrum(sig, t):
    f = fftpack.rfftfreq(sig.size, d=t[1] - t[0])
    y = fftpack.rfft(sig)
    return f, np.abs(y)


def bandpass(f, sig, min_freq, max_freq):
    sig[np.logical_or(f < min_freq, f > max_freq)] = 0
    return sig


# decompose into frequency domain
freq, spec = spectrum(signal, time)

# apply bandpass filter to frequencies, then convert back to filtered signal
signal_filtered = fftpack.irfft(bandpass(freq, spec, 2.5, 3))

plt.plot(time, signal)
plt.title("cos(4 * pi * time) + cos(6 * pi * time) + cos(8 * pi * time)")
plt.show()

plt.plot(time, signal_filtered)
plt.title("Signal filtered to just: cos(6 * pi * time)")
plt.show()
