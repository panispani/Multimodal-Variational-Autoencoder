import torch
import torch.nn as nn
from torchvision import datasets, transforms

#### Everything is as described in Appendix A of the paper ####

# PARAMETERS
IMAGESIZE = 784
N_LABELS = 10
EMBED_DIMS = 10
Z_DIMS = 64 # latent dimensions
BATCH_SIZE = 100
EPOCHS = 500
ANNEAL_EPOCHS = 200 # Epochs to anneal for KL
L_RATE = 1E-3
BATCH_LOGGING_INTERVAL = 2
MULTIPLIER_IMAGE = 1
MULTIPLIER_LABEL = 50
# Load train and test data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE,
    shuffle=False)
N_MINI_BATCHES = len(train_loader)

# Encoders and Decoders
class LabelEncoder(nn.Module):
    def __init__(self):
        super(LabelEncoder, self).__init__()
        self.fc1 = nn.Embedding(N_LABELS, N_LABELS)
        self.fc2 = nn.Linear(N_LABELS, 512)
        self.fc_means = nn.Linear(512, Z_DIMS)
        self.fc_logvar = nn.Linear(512, Z_DIMS)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        return self.fc_means(h), self.fc_logvar(h)

class LabelDecoder(nn.Module):
    def __init__(self):
        super(LabelDecoder, self).__init__()
        self.fc1 = nn.Linear(Z_DIMS, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, N_LABELS)

    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        return self.fc_out(h)

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.fc1 = nn.Linear(IMAGESIZE, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_means = nn.Linear(512, Z_DIMS)
        self.fc_logvar = nn.Linear(512, Z_DIMS)

    def forward(self, x):
        h = self.fc1(x.view(-1, IMAGESIZE))
        h = self.fc2(h)
        return self.fc_means(h), self.fc_logvar(h)

class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()
        self.fc1 = nn.Linear(Z_DIMS, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, IMAGESIZE)

    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        return self.fc_out(h)
