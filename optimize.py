import optuna
import torch
import torch.nn as nn

from model import TransformerClassifier
from data_load import CustomDataLoader


data_root = "train_data"
test_root = "test_data"
batch_size = 32

train_loader, val_loader, test_loader, sequence_length, feature_size = CustomDataLoader.create_data_loaders(data_root, test_root, batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):

    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 8)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    out_channels = trial.suggest_int('out_channels', 16, 64)
    kernel_size = trial.suggest_int('kernel_size', 3, 7)

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
            # Move data to GPU
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels.long())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for val_data, val_labels in val_loader:
            # Move validation data to GPU
            val_data, val_labels = val_data.to(device), val_labels.to(device)

            val_outputs = model(val_data)
            _, predicted = torch.max(val_outputs.data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels.long()).sum().item()

        accuracy = correct / total
        print(f'{hidden_size}_{num_heads}_{num_encoder_layers}_{dropout}_{out_channels}_{kernel_size}_Validation Accuracy: {100 * accuracy:.2f}%')
    
    torch.save(model.state_dict(), 'save_models/'+f'model_transformer_{hidden_size}_{num_heads}_{num_encoder_layers}_{dropout}_{out_channels}_{kernel_size}.pth')
    
    return accuracy


study = optuna.create_study(direction='maximize') 
study.optimize(objective, n_trials=50)  

best_params = study.best_params
best_value = study.best_value

print(f"Best hyperparameters: {best_params}")
print(f"Best value (accuracy): {best_value}")
