import numpy as np

def gauss_sig(A, T0, F0, PHI, P, T, fs):
    """
    Creates a signal of transient and oscillating components with Gaussian envelopes.
    
    Parameters
    ----------
    A : np.array/list/number - Amplitudes of transients.
    T0 : np.array/list/number - Time centres of transients.
    F0 : np.array/list/number - Frequency centres of transients.
    PHI : np.array/list/number - Phase of transients.
    P : np.array/list/number - Scaling parameters of Gaussian envelopes for transients.
    T : float/int - Measurement time/end time.
    fs : float/int - Sampling frequency.
    

    Returns
    -------
    x : np.array - Real-valued signal.
    t : np.array - Time indices.

    Implemented by: Isabella Reinhold, Lund University
    """
    
    N = int(np.rint(T*fs)) + 1
    t = np.array(range(N), dtype=float) / fs
    F0 = np.array(F0)
    T0 = np.array(T0)
    P = np.array(P)
    PHI = np.array(PHI)
    A = np.array(A)
    K = F0.size

    if K == 1:
        x = A * (np.cos(2*np.pi*F0*t + PHI) * (np.exp(-np.square(t-T0) / (2 *np.square(P)))))
    else:
        x = np.zeros((N), dtype=float)
        for k in range(K):
            x = x + A[k] * (np.cos(2*np.pi*F0[k]*t + PHI[k]) * (np.exp(-np.square(t-T0[k]) / (2 *np.square(P[k])))))
    
    return x, t
