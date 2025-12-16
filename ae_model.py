import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim: int = 16, hidden_dim: int = 128):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.z_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())

        self.decoder_mean = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Softmax(dim=-1))
        self.decoder_dispersion = nn.Parameter(torch.randn(input_dim))
        self.decoder_dropout = nn.Linear(hidden_dim, input_dim)  # scvi ZINB expects logits

    def encode(self, x):
        if x.max() > 20:
            print(
                f"warning: input to encode has max: {x.max().item():.2f} but the model expects log1p transformed data"
            )
        h = self.encoder(x)
        z = self.z_layer(h)
        return z

    def decode(self, z):
        """returns mean proportions"""
        h_decoded = self.decoder(z)
        mean = self.decoder_mean(h_decoded)
        return mean

    def forward(self, x):
        h = self.encoder(x)
        z = self.z_layer(h)

        h_decoded = self.decoder(z)

        mean = self.decoder_mean(h_decoded)
        dispersion = torch.exp(self.decoder_dispersion)
        dropout = self.decoder_dropout(h_decoded)

        return (mean, dispersion, dropout, z)
