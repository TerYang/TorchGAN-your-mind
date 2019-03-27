# the generator network
# input sample size : 1 * 1600

import torch.nn as nn
import torch.nn.functional as F

class Generator2(nn.Module):
    def __init__(self):
        super(Generator2,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),  # 1 * 400 ==> 64 * 400
            nn.BatchNorm1d(64, momentum=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 64 * 400 ==> 64 * 200

            # nn.Conv1d(64, 128, kernel_size=15, padding=7),  # 64 * 200 ==> 128 * 200
            # nn.BatchNorm1d(128, momentum=0.5),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),  # 128 *200 ==> 128 * 100

        )

        self.decoder = nn.Sequential(
            # nn.Upsample(scale_factor=2),  # 128 * 100 ==> 128 * 200
            # nn.Conv1d(128, 64, kernel_size=15, padding=7),  # 128 * 200 ==> 64 * 200
            # nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 64 * 200 ==> 64 * 400
            nn.Conv1d(64, 1, kernel_size=15, padding=7),  # 64 * 400 ==> 1 * 400
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv1d(1, 64 , kernel_size = 15 , padding= 7),   # 1 * 100 ==> 64 * 100
        #     # nn.BatchNorm1d(32, momentum=0.5),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size = 2 ),                     # 64 * 100 ==> 64 * 50
        #
        #     nn.Conv1d(64, 128 , kernel_size= 15 , padding= 7),  # 64 * 50 ==> 128 * 50
        #     # nn.BatchNorm1d(64, momentum=0.5),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size= 2),                       # 128 *50 ==> 128 * 25
        #     # nn.Conv1d(128, 64, kernel_size=15, padding=7),
        #     # nn.ReLU(),
        #     # nn.Conv1d(64, 1, kernel_size=15, padding=7)
        #
        # )
        #
        # self.decoder = nn.Sequential(
        #     nn.Upsample(scale_factor = 2),                      # 128 * 25 ==> 128 * 50
        #     nn.Conv1d(128, 64, kernel_size= 15 ,  padding=7),   # 128 * 50 ==> 64 * 50
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor = 2),                      # 64 * 50 ==> 64 * 100
        #     nn.Conv1d(64 , 1, kernel_size= 15 ,padding= 7),     # 64 * 100 ==> 1 * 100
        #
        # )


    def forward(self,x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return F.sigmoid(x2)