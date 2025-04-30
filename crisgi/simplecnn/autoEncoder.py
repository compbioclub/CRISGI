from torch import nn



class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 4, 3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        en = self.encoder(x)
        de = self.decoder(en)
        return en, de
    

