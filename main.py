import torch
import torch.nn as nn

from data_load import CustomDataLoader
from model import TransformerClassifier
'''
Best parameter
'hidden_size': 161, 'num_heads': 8, 'num_encoder_layers': 4, 'dropout': 0.1416134672882693, 'out_channels': 37, 'kernel_size': 3
'''

data_root = "train_data"
test_root = "test_data"
batch_size = 32

train_loader, val_loader, test_loader, sequence_length, feature_size = CustomDataLoader.create_data_loaders(data_root, test_root, batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size=161
num_heads=8
num_encoder_layers=4
dropout=0.1416134672882693
out_channels=37
kernel_size=3

if hidden_size % num_heads != 0:
    hidden_size = (hidden_size // num_heads) * num_heads

model = TransformerClassifier(in_channels=sequence_length, 
                              feature_size=feature_size,
                              hidden_size=hidden_size,
                              num_heads=num_heads,
                              num_encoder_layers=num_encoder_layers,
                              dropout=dropout,
                              out_channels=out_channels,
                              kernel_size=kernel_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

num_epochs = 100
model.train()
for epoch in range(num_epochs):
    for batch_data, batch_labels in train_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for val_data, val_labels in val_loader:
            val_data, val_labels = val_data.to(device), val_labels.to(device)

            val_outputs = model(val_data)
            _, predicted = torch.max(val_outputs.data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels.long()).sum().item()

        accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {100 * accuracy:.2f}%')
    model.train()

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'hidden_size': hidden_size,
    'num_heads': num_heads,
    'num_encoder_layers': num_encoder_layers,
    'dropout': dropout,
    'out_channels': out_channels,
    'kernel_size': kernel_size,
}, 'save_models/model_transformer_v_final.pth')
