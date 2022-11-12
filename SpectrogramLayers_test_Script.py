"""
Example on how to use classes in SpectogramLayers.
Also requires help functions in gauss_sig - creates signal data, and
binary_class_helpfn - for training loop and inference.
"""
import torch
import math
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from SpectrogramLayers import SpecDataset, mtSpecLayer
from gauss_sig import gauss_sig
from binary_class_helpfn import train_step, val_step, make_predictions


# --- HYPERPARAMATERS ---
K_IN = 10
K_OUT = True
F_SIZE = 64
T_FACT = 4
NBR_SAMPLES = 1200
BATCH_SIZE = 32
LR = 0.001
epochs = 20

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --- DATA ---
# Generate data
sig_length = 199
gauss_scaling = 8
P = [gauss_scaling, gauss_scaling]
fs = 1
X = []
y = []
for n in range(int(NBR_SAMPLES/2)):
    # Two trainsient signals, one or two components
    amplitudes = [1+0.5*np.random.rand(1), 1+0.5*np.random.rand(1)]
    t0 = 30+140*np.random.rand(1)
    delta_t = 2*gauss_scaling + 2*gauss_scaling*np.random.rand(1)
    time_centres = [t0, t0+delta_t]
    f0 = 0.1+0.3*np.random.rand(1)
    frequency_centres = [f0, f0]
    phases = [math.pi*np.random.rand(1), math.pi*np.random.rand(1)]
    x1, t2 = gauss_sig(amplitudes[0], time_centres[0], frequency_centres[0], phases[0], P[0], sig_length, fs)
    x2, t2 = gauss_sig(amplitudes, time_centres, frequency_centres, phases, P, sig_length, fs)
    # Add white Gaussian noise
    noise = 0.5*np.random.randn(sig_length+1)
    x1, x2 = x1 + noise, x2 + noise
    X.append([x1, x2])
    y.append([0, 1])
X =  np.concatenate(X)
y =  np.concatenate(y)

# Calculate spectrograms
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
train_data = SpecDataset(row_data=X_train,
                         labels=y_train,
                         classes=["One", "Two"],
                         K=K_IN,
                         p=gauss_scaling,
                         fs=fs,
                         f_size=F_SIZE,
                         t_fact=T_FACT,
                         label_dtype=torch.float)
val_data = SpecDataset(row_data=X_val,
                       labels=y_val,
                       classes=["One", "Two"],
                       K=K_IN,
                       p=gauss_scaling,
                       fs=fs,
                       f_size=F_SIZE,
                       t_fact=T_FACT,
                       label_dtype=torch.float)
T_SIZE = train_data.tsize

# Visualise time signals
plt.close("all")
fig1 = plt.figure(1)
rows, cols = 4, 4
idx = torch.randint(0, len(train_data), size=[rows*cols])
for k in range(1, rows*cols+1):
    fig1.add_subplot(rows, cols, k)
    plt.plot(X_train[idx[k-1]])
    plt.title(train_data.classes[int(train_data.targets[idx[k-1]])], fontsize=8)
    plt.axis(False)

# Visualise spectrograms
fig2 = plt.figure(2)
for k in range(1, rows*cols+1):
    fig2.add_subplot(rows, cols, k)
    plt.pcolor(train_data.data[idx[k-1], 0, :, :].T)
    plt.title(train_data.classes[int(train_data.targets[idx[k-1]])], fontsize=8)
    plt.axis(False)

# Dataloader
train_dl = DataLoader(train_data,
                     batch_size=BATCH_SIZE,
                     shuffle=True)
val_dl = DataLoader(val_data,
                     batch_size=BATCH_SIZE,
                     shuffle=False)


# --- MODEL ---
# Simple network example
class testSpec(nn.Module):
    def __init__(self, K, t_size, f_size, K_out):
        super().__init__()
        
        # Number of out channels from mtSpecLayer
        out_ch = 1
        if K_out:
            out_ch = K
        
        self.speclayer = mtSpecLayer(K, K_out)
        
        self.drop = nn.Dropout(0.3)
        
        self.cnnblock = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(in_channels=out_ch,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
            )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=int(32*int(t_size/2/2)*int(f_size/2/2)),
                      out_features=1),
            )
    def forward(self, x):
        x = self.speclayer(x)
        x = self.cnnblock(x)
        x = self.drop(x)
        x = self.classifier(x)
        return x

# Initialise model
model = testSpec(K_IN, T_SIZE, F_SIZE, K_OUT)

# Loss function and optimiser
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(params=model.parameters(), lr=LR)


# --- TRAINING ---
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch+1}\n------")
    # Training
    train_step(model, train_dl, loss_fn, opt, DEV)

    # Validation
    val_step(model, val_dl, loss_fn, opt, DEV)


# --- CONFUSION MATRIX ---
y_preds = make_predictions(model, val_dl)
y_targets = val_dl.dataset.targets.type(torch.int)
confmat = ConfusionMatrix(num_classes=2)

confmat_tensor = confmat(preds=y_preds,
                         target=y_targets)
fig3, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=val_data.classes,
    figsize=(10, 9)
    )
plt.title("Confusion matrix, torchmetrics")

