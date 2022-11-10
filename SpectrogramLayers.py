import torch
import math

# Class that turns signals into K orthogonal spectrograms
class SpecDataset(torch.utils.data.Dataset):
    """
    Creates spectrogram data set from 1D time signals. Data set can be used with
    e.g. torch.utils.data.DataLoader. Output data a tensor of dim:
    nbr_samples x K x ceil(signal length/t_fact) x f_size.

    Parameters
    ----------
    row_data : list or np.array - 1D time signal as list elements or row vectors
    in np.array, dim: nbr_samples x signal_length.
    labels : list or np.array - Labels, dim: nbr_samples.
    classes : list - List of strings with class names, dim: nbr_classes.
    K : int - Number of spectrograms. 
    p : float - Scaling parameter for Gaussian function used for windows.
    fs : float - Sampling frequency.
    f_size : int - Output dimension for frequency axis.
    t_fact : int - Downscaling factor for time axis, t_fact = 1 => no downscaling.
    label_dtype : dtype - Data type of target/label tensor, need will depend on
    loss function.

    Implemented by: Isabella Reinhold, Lund University
    """
    def __init__(self,
                 row_data,
                 labels,
                 classes: list,
                 K: int,
                 p: float,
                 fs: float,
                 f_size: int,
                 t_fact: int,
                 label_dtype = torch.long):
        
        # Make singals and tagets/labels into tensors
        X, y = torch.tensor(row_data, dtype=torch.float), torch.tensor(labels, dtype=label_dtype)
        
        # Prepare signals
        X -= torch.mean(X, dim=1, keepdim=True)
        X /= torch.linalg.norm(X, dim=1, keepdim=True)
        
        # Calculate spectrograms
        nbr_samples, X_length = X.shape
        t_size = int(math.ceil(X_length/t_fact))
        Win = _hermitewin(K, p*fs, X_length)
        S = torch.empty(size=(nbr_samples, K, t_size, f_size), dtype=torch.float)
        for sample in range(nbr_samples):
            S[sample, :, :, :] = _spect(X[sample, :], Win, K, f_size, t_fact)
        
        # Make sure dimensions match
        assert all(S.size(0) == tensor.size(0) for tensor in (S, y))
        
        # Attributes
        self.data = S
        self.targets = y
        self.labels = y
        self.classes = classes
        self.K = K
        self.fsize = f_size
        self.tsize = t_size
    
    # Necessary functions
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
        
    def __len__(self):
        return self.data.size(0)

# Class to work as initial layer in an ANN where the data is K channels of orthogonal spectrograms
class mtSpecLayer(torch.nn.Module):
    """
    ANN layer for K channels of orthogonal spectrogram data. Used to optimise
    weights of the K spectrograms. Output is either K channels (K_out = True)
    or one channel as a weighted sum (K_out = False).

    Parameters
    ----------
    K : int - Number of input spectrograms/channels.
    K_out : bool - True - K channels out, False - 1 channel out (weighted sum of
    input spectrograms/channels.)
    
    Implemented by: Isabella Reinhold, Lund University
    """
    def __init__(self,
                 K: int,
                 K_out: bool = True):
        super().__init__()
        
        # Weights
        self.w = torch.nn.Parameter(torch.ones(K,
                                               dtype=torch.float,
                                               requires_grad=True) / K)
        # Number of in and out channels
        self.K = K
        if K_out:
            self.out_channels = K
        else:
            self.out_channels = 1
    
    def forward(self, x):
        # Multiply by weights
        x = self.w.view((1, self.K, 1, 1)) * x
        
        # Weighted sum if K_out = False
        if self.out_channels == 1:
            x = torch.sum(x, dim=1, keepdim=True)
        
        return x

# --- Help functions ---
# Hermite windows
def _hermitewin(K, p, in_shape):
    
    # Time vector, centre = 0
    M = min(int(12 * p), in_shape)
    tvect = torch.arange(start=-int(M/2), end=int(M/2), dtype=torch.int)
    M = len(tvect)
    
    # Polynomials (physicists')
    He = torch.ones(K, M, dtype=torch.float)
    if K > 1:
        He[1, :] = 2 / p * tvect
        for k in range(2, K):
            He[k, :] = 2 / p * (He[k-1, :] * tvect) - 2 * (k-1) * He[k-2, :]     
    
    # Unit energy windows
    Win = He * torch.exp(-torch.square(tvect) / (2 * (p ** 2)))
    Win = Win / torch.linalg.norm(Win, dim=1, keepdim=True)
    
    return Win

# Spectrograms
def _spect(x, Win, K, f_size, t_fact):
    
    # Window length
    M = Win.shape[1]
    
    # Zero-pad signal
    if x.shape[0] % 2 != 0:
        x = torch.cat((torch.zeros(int(M/2), dtype=torch.float),
                            x,
                            torch.zeros(int(M/2 + 1), dtype=torch.float)), dim=0)
    else:
        x = torch.cat((torch.zeros(int(M/2), dtype=torch.float),
                            x,
                            torch.zeros(int(M/2), dtype=torch.float)), dim=0)
    N = x.shape[0]
    
    # STFTs (assumes real valued signal)
    nfft = int(f_size * 2)
    E = math.sqrt(nfft / t_fact)
    ind = -1
    t_size = int(math.ceil((N-M)/t_fact))
    F = torch.empty((K, t_size, f_size), dtype=torch.cfloat)
    for j in range(0, N-M, t_fact):
        ind = ind + 1
        x_step = x[range(j, j+M)]
        F_temp = torch.fft.rfft(Win * x_step, n=nfft, dim=1) / E
        F[:, ind, :] = F_temp[:, :f_size]
       
    # Spectrograms
    S = torch.square(abs(F))
    
    return S
