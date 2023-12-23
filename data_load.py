import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, data_array, labels_array):
        self.data_array = data_array
        self.labels_array = labels_array

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        return torch.Tensor(self.data_array[idx]), torch.Tensor([self.labels_array[idx]])

def collate_fn(batch):
    data, labels = zip(*batch)
    return torch.stack(data), torch.cat(labels)

class CustomDataLoader:
    @staticmethod
    def create_data_loaders(data_root, test_root, batch_size):
        data_list, labels_list, _, min_sequence_length_train = CustomDataLoader.load_data_train(data_root)
        test_list, _, min_sequence_length_test = CustomDataLoader.load_data_test(test_root)

        min_sequence_length = min(min_sequence_length_train, min_sequence_length_test)
        
        for i in range(len(data_list)):
            current_sequence_length = data_list[i].shape[0]
            if current_sequence_length > min_sequence_length:
                data_list[i] = data_list[i][:min_sequence_length, :]

        for i in range(len(test_list)):
            current_sequence_length = test_list[i].shape[0]
            if current_sequence_length > min_sequence_length:
                test_list[i] = test_list[i][:min_sequence_length, :]

        data_array = np.array(data_list)
        labels_array = np.array(labels_list)

        test_array = np.array(test_list)
        test_tensor = torch.tensor(test_array)

        # Create a dataset
        dataset = CustomDataset(data_array, labels_array)

        # Split the dataset into training and validation sets
        total_samples = len(dataset)
        train_size = int(0.8 * total_samples)
        val_size = total_samples - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
        test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader, test_loader, min_sequence_length, test_tensor.shape[2]

    @staticmethod
    def load_data_train(data_root):
        data_list = []
        labels_list = []
        max_sequence_length = 0
        min_sequence_length = 10000

        for language_folder in os.listdir(data_root):
            language_path = os.path.join(data_root, language_folder)

            if os.path.isdir(language_path):
                for npy_file in os.listdir(language_path):
                    if npy_file.endswith(".npy"):
                        npy_path = os.path.join(language_path, npy_file)
                        data = np.load(npy_path)
                        max_sequence_length = max(max_sequence_length, data.shape[0])
                        min_sequence_length = min(min_sequence_length, data.shape[0])
                        data_list.append(data)
                        labels_list.append(int(language_folder.split('_')[-1]))

        return data_list, labels_list, max_sequence_length, min_sequence_length
    
    @staticmethod
    def load_data_test(test_root):
        test_list = []
        max_sequence_length = 0
        min_sequence_length = 10000

        for npy_file in os.listdir(test_root):
            if npy_file.endswith(".npy"):
                npy_path = os.path.join(test_root, npy_file)
                test = np.load(npy_path)
                max_sequence_length = max(max_sequence_length, test.shape[0])
                min_sequence_length = min(min_sequence_length, test.shape[0])
                test_list.append(test)

        return test_list, max_sequence_length, min_sequence_length

