import numpy as np
import math

# --- Main function ---
def mtspec(x: np.array,
                  K: int,
                  fs: float,
                  p: float,
                  p_option: str = "sigma",
                  nfft: int = 1024,
                  t_fact: int = 1):
    """
    Calculates the multitaper spectrogram. Uses overlapping Hermite windows. 
    The multitaper spectrogram will be a tensor of dim
    nfft/2 x ceil(signal length/t_fact).
    
    Parameters
    ----------
    x : np.array - Real-valued signal.
    K : int - Number of windows.
    fs : float - Sampling frequency.
    p : float - Window scaling or length parameter, see p_option.
    p_option: string - Option for p, defined as either:
    "sigma" - scaling paramater of Gaussian function,
    "fwhm" - length according to full width at half maximum,
    "p96" - length according to 96% of energy is full width.
    nfft : int - Number of frequency points evaluated in fft.
    t_fact : int - Downscaling factor for time axis, t_fact = 1 => no downscaling.

    Returns
    -------
    MTS : np.array - The multitaper spectrogram.
    f : np. array - Frequency indices.
    t : np.array - Time indices.
    
    Implemented by: Isabella Reinhold, Lund University
    """
    
    # Determine window scaling from length
    if p_option.lower() == "fwhm":
        p = p / (2 * math.sqrt(2 * math.log(2)))
    elif p_option.lower() == "p96":
        p = p / 4
    # Convert scaling to samples
    p = p * fs 
    
    # Hermite windows
    N = x.size
    Win = _hermitewin(K, p, N)
    
    # Spectrogram
    MTS = _spect(x, Win, nfft, t_fact)
   
    # Even/odd
    if N % 2 != 0:
        MTS = MTS[:, :N]
    
    # Frequency and time vectors
    f = np.array(range(int(nfft/2)), dtype=float) * fs / nfft
    t = np.array(range(int(math.ceil(N/t_fact))), dtype=float) / fs * t_fact
    
    return MTS, f, t

# --- Help functions ---
# Hermite windows
def _hermitewin(K, p, N):
    
    # Time vector, centre = 0
    M = min(int(12 * p), N)
    tvect = np.array(range(-int(M/2), int(M/2)), dtype=int)
    M = tvect.size
    
    # Polynomials (physicists')
    He = np.ones((K, M), dtype=float)
    if K > 1:
        He[1, :] = 2 / p * tvect
        for k in range(2, K):
            He[k, :] = 2 / p * (He[k-1, :] * tvect) - 2 * (k-1) * He[k-2, :]     
    
    # Unit energy windows
    Win = He * np.exp(-np.square(tvect) / (2 * np.square(p)))
    Win /= np.reshape(np.linalg.norm(Win, axis=1), (K, 1))
    
    return Win

# Multitaper spectrogram
def _spect(x, Win, nfft, t_fact):
    
    # Number of windows and their length
    K, M = np.shape(Win)
    
    # Zero-pad signal
    if x.size % 2 != 0:
        x = np.concatenate((np.zeros(int(M/2), dtype=float),
                            x,
                            np.zeros(int(M/2 + 1), dtype=float)))
    else:
        x = np.concatenate((np.zeros(int(M/2), dtype=float),
                            x,
                            np.zeros(int(M/2), dtype=float)))
    N = x.size
    
    # STFTs (assumes real valued signal)
    E = math.sqrt(nfft * t_fact)
    ind = -1
    nfft2 = int(nfft/2)
    ind_max = int(math.ceil((N-M)/t_fact)) 
    F = np.empty((K, ind_max, nfft2), dtype=complex)
    for j in range(0, N-M, t_fact):
        ind = ind + 1
        x_step = x[range(j, j+M)]
        F_temp = np.fft.rfft(Win * x_step, n=nfft, axis=1) / E
        F[:, ind, :] = F_temp[:, :nfft2]
       
    # Multitaper spectrogram
    MTS = np.mean(np.square(abs(F)), axis=0)
    
    return MTS.T
