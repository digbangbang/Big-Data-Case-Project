import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, in_channels, feature_size, hidden_size, num_heads, num_encoder_layers, dropout, out_channels, kernel_size):
        super(TransformerClassifier, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.relu = nn.SELU()
        self.embedding = nn.Linear(out_channels * (feature_size - kernel_size + 1), hidden_size)

        self.transformer_encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layers, num_layers=num_encoder_layers
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.relu(self.conv1d(x))
        x = x.view(x.shape[0], -1)
        x = self.embedding(x)
        x = x.view(-1, x.shape[0], x.shape[1])
        output = self.transformer_encoder(x)
        output = output.view(output.shape[1], -1)
        output = self.fc(output)

        return output

