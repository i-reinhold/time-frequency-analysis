import numpy as np
import math

# --- Main function ---
def mtreassignedspec(x: np.array,
                     K: int,
                     fs: float,
                     p: float,
                     p_option: str = "sigma",
                     nfft: int = 1024,
                     t_fact: int = 1,
                     e: float = 0.001):
    """
    Calculates the multitaper reassigned spectrogram and multitaper spectrogram,
    designed for transient oscillating signals with Gaussian envelopes. Uses
    overlapping Hermite windows. The multitaper reassigned spectrogram will be
    a matrix of dim nfft/2 x ceil(signal length/t_fact).
    Implemented according to method by Reinhold and Sandsten (2022) doi:
    10.1016/j.sigpro.2022.108570
    
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
    e : float - Energy threshold (less than 1).

    Returns
    -------
    MTRS : np.array - The multitaper reassigned spectrogram.
    MTS : np.array - The multitaper spectrogram.
    f : np. array - Frequency indices.
    t : np.array - Time indices.
    
    Notes
    -------
    Set K = 1 for the scaled reassigned spectrogram, Sandsten and Brynolfsson
    (2015) doi: 10.1109/LSP.2014.2350030
    
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
    Win, tWin, dWin = _hermitewin(K, p, N)
    
    # Spectrogram and STFTs
    S, F, tF, dF = _stft(x, Win, tWin, dWin, nfft, t_fact)
    
    # Reassignment
    MTRS, MTS = _reassign(S, F, tF, dF, p, t_fact, e)
    
    # Even/odd
    if N % 2 != 0:
        MTS = MTS[:, :N]
        MTRS = MTRS[:, :N]
    
    # Frequency and time vectors
    f = np.array(range(int(nfft/2)), dtype=float) * fs / nfft
    t = np.array(range(int(math.ceil(N/t_fact))), dtype=float) / fs * t_fact
    
    return MTRS, MTS, f, t

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
    
    # All windows
    wfun = np.exp(-np.square(tvect) / (2 * np.square(p))) / math.sqrt(math.sqrt(math.pi) * p)
    Win = He * wfun
    tWin = Win * tvect
    dWin = np.empty((K, M), dtype=float)
    dWin[0, :] = -((He[0, :] * tvect) * wfun) / np.square(p)
    for k in range(1, K):
        dWin[k, :] = 2 * k / p * (He[k-1, :] * wfun) - ((He[k, :] * tvect) * wfun) / np.square(p)
    
    # Unit energy windows
    E = np.reshape(np.linalg.norm(Win, axis=1), (K, 1))
    Win /= E
    tWin /= E
    dWin /= E

    return Win, tWin, dWin

# STFTs
def _stft(x, Win, tWin, dWin, nfft, t_fact):
    
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
    E = math.sqrt(nfft / t_fact)
    ind = -1
    nfft2 = int(nfft/2)
    ind_max = int(math.ceil((N-M)/t_fact)) 
    F = np.empty((K, ind_max, nfft2), dtype=complex)
    tF = np.empty((K, ind_max, nfft2), dtype=complex)
    dF = np.empty((K, ind_max, nfft2), dtype=complex)
    for j in range(0, N-M, t_fact):
        ind = ind + 1
        x_step = x[range(j, j+M)]
        F_temp = np.fft.rfft(Win * x_step, n=nfft, axis=1) / E
        F[:, ind, :] = F_temp[:, :nfft2]
        tF_temp = np.fft.rfft(tWin * x_step, n=nfft, axis=1) / E
        tF[:, ind, :] = tF_temp[:, :nfft2]
        dF_temp = np.fft.rfft(dWin * x_step, n=nfft, axis=1) / E
        dF[:, ind, :] = dF_temp[:, :nfft2]
       
    # All spectrograms
    S = np.square(abs(F))
    
    return S, F, tF, dF

# Reassignment
def _reassign(S, F, tF, dF, p, t_fact, e):
    
    # Number of windows, time bins, frequency bins
    K, M, N = S.shape
    
    # K reassignment vectors
    meshf, mesht = np.meshgrid(range(N), range(M))
    # Division by 0 will give no reassignment (tmat0 = 0 or fmat0 = 0)
    tmat0 = 1 / t_fact * np.real(np.divide(
        tF - np.square(p) * dF,
        F, out=np.zeros((K, M, N), dtype=complex), where=F!=0)
        )
    fmat0 = N / math.pi * np.imag(
        np.divide(dF - (tF / np.square(p)),
                  F,
                  out=np.zeros((K, M, N), dtype=complex), where=F!=0)
        )
    tmat = mesht + tmat0
    fmat = meshf - fmat0
 
    # Energy threshold and within bounds reassignment
    Se = np.reshape(e * np.amax(S, axis=(1,2)), (K, 1, 1))
    S_check = (S > Se) * (tmat >= 0) * (tmat <= M-1) * (fmat >= 0) * (fmat <= N-1)
    
    # Average reassignment vectors
    S_check_sum = np.sum(S_check, axis=0)
    # A total of 0 valid reassignment vectors == division by 0, will yield invalid (negative) average reassignment vector
    mtmat = np.rint(np.divide(
        np.sum(S_check * tmat, axis=0),
        S_check_sum,
        out=-np.ones((M, N), dtype=float), where=S_check_sum!=0)
        )
    mfmat = np.rint(np.divide(
        np.sum(S_check * fmat, axis=0),
        S_check_sum,
        out=-np.ones((M, N), dtype=float), where=S_check_sum!=0)
        )
    
    # Multitaper spectrogram
    MTS = np.mean(S, axis=0)
    
    # New energy threshold
    MTSe = e * np.amax(MTS, axis=(0,1))
                
    # Reassignment
    MTRS = np.zeros((M, N), dtype=float)
    for m in range(M):
        for n in range(N):
            if MTS[m, n] > MTSe and mtmat[m, n] >= 0 and mtmat[m, n] <= M-1 and mfmat[m, n] >= 0 and mfmat[m, n] <= N-1:
                new_t = int(mtmat[m, n])
                new_f = int(mfmat[m, n])
                MTRS[new_t, new_f] = MTRS[new_t, new_f] + MTS[m, n]
    
    return MTRS.T, MTS.T