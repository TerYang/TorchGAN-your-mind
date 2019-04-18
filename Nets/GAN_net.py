# the generator network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.encoder = nn.Sequential(
            # nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),  # keep dims conv,shape: 64*22*1 to 64*22*64
            # nn.ReLU(),
            # # nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),  # scale down dims conv,shape: 32*11*64 to 32*11*128
            # nn.ReLU(),
            # # nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),  # scale down dims conv,shape: 16*6*128 to 16*6*256
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),  # cale down dims conv,shape: 16*6*256 to 8*3*256
            # nn.ReLU(),
            # # nn.Softmax2d
            # # nn.LeakyReLU(),
            # # nn.Sigmoid()

            nn.Conv2d(1, 16, kernel_size=(4,2), stride=2, padding=1),  # keep dims conv,shape: 64*21 to 32, 11
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(4,2), stride=2, padding=1),  #  32, 11 to 16, 6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#16, 6 -> 8, 3
            nn.Conv2d(32, 64, kernel_size=(4,2), stride=2, padding=1),  #  8, 3 to  4, 2
            nn.ReLU(),
        )

        self.L_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            # nn.Sigmoid()# added
        )
    def forward(self,x):
        x1 = self.encoder(x)
        x2 = x1.view(x1.size(0), 512)
        x3 = self.L_layer(x2)
        return torch.sigmoid(x3)