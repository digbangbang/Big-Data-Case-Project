import torch

from model import TransformerClassifier
from data_load import CustomDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = "train_data"
test_root = "test_data"
batch_size = 32

_, _, test_loader, sequence_length, feature_size = CustomDataLoader.create_data_loaders(data_root, test_root, batch_size)


checkpoint = torch.load('save_models/model_transformer_v_final.pth')

model = TransformerClassifier(in_channels=sequence_length,
                              feature_size=feature_size,
                              hidden_size=checkpoint['hidden_size'],
                              num_heads=checkpoint['num_heads'],
                              num_encoder_layers=checkpoint['num_encoder_layers'],
                              dropout=checkpoint['dropout'],
                              out_channels=checkpoint['out_channels'],
                              kernel_size=checkpoint['kernel_size']).to(device)

model.load_state_dict(checkpoint['model_state_dict'])

test_label = []
model.eval()
with torch.no_grad():
    for test_data in test_loader:
        # Move validation data to GPU
        test_data = test_data.to(device)

        test_outputs = model(test_data)
        _, predicted = torch.max(test_outputs.data, 1)
        
        test_label.extend(predicted.tolist())


import pandas as pd

test = pd.read_csv('test.csv')
test['label'] = test_label
test.to_csv('test1.csv', index=False)