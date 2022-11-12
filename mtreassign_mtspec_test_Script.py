"""
Example on how to use the multitaper reassigned spectrogram and the multitaper
spectrogram.
Also requires help function in gauss_sig - creates transient signal.
"""

from mtreassignedspec import mtreassignedspec
from mtspec import mtspec
from gauss_sig import gauss_sig

import matplotlib.pyplot as plt
import numpy as np

# Parameters
sig_length = 249.5
K = 3
p = 7
fs = 2
nfft = 256
t_fact = 2


# Multicomponent transient signal
amplitudes = [0.9, 1, 1]
time_centres = [70, 70, 130]
frequency_centres = [0.4, 0.6, 0.6]
phases = [0, np.pi/4, np.pi/2]
P = [p, p, p]
x, t_x = gauss_sig(amplitudes, time_centres, frequency_centres, phases, P, sig_length, fs)


# Multitaper reassigned spectrogram
MTRS, MTS, f, t = mtreassignedspec(x=x, K=K, fs=fs, p=p, p_option="std", nfft=nfft, t_fact=t_fact)

# Multitaper spectrogram
MTS, f, t = mtspec(x=x, K=K, fs=fs, p=p, nfft=nfft, t_fact=t_fact)


# Plots
fig, ax = plt.subplots(3, figsize=(6, 8))
ax[0].plot(t_x, x)
ax[1].pcolor(t, f, MTRS)
ax[2].pcolor(t, f, MTS)
