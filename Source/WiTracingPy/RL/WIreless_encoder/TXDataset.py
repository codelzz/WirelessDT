from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


# Define the dataset class
def tolist(df):
    return df.tolist()

class TXDataset(Dataset):
    def __init__(self, annotations_file, tokenizer, pad_value='-', pad_length=30, device='cuda'):
        self.data = pd.read_csv(annotations_file)
        self.data = self.data.groupby(['timestamp', 'x', 'y', 'z'])['rssi', 'tx'].agg(tolist).reset_index()
        self.tokenizer = tokenizer
        self.pad_value = pad_value
        self.pad_length = pad_length
        self.pad_string = pad_value * pad_length

    def __len__(self):
        return len(self.data)

    def pad_list(self, rssi_list, target_length=20, padding_value=-255):
        padded_list = rssi_list.copy()
        while len(padded_list) < target_length:
            padded_list.append(padding_value)
        padded_list = padded_list[:20]
        return padded_list

    def __getitem__(self, idx):
        # Get the TXName string from the data
        tx_names = self.data.iloc[idx]['tx']
        tx_rssis = self.data.iloc[idx]['rssi']

        # Pad each string with spaces to make them have equal length
        # padded_tx_names = [s.ljust(self.pad_length, self.pad_value) for s in tx_names]
        padded_tx_rssis = self.pad_list(tx_rssis)

        while len(tx_names) < 20:
            tx_names.append(self.pad_value)
        if len(tx_names) >= 20:
            tx_names = tx_names[:20]
        tx_names = ' '.join(tx_names)
        label = (self.data.iloc[idx]['x'], self.data.iloc[idx]['y'], self.data.iloc[idx]['z'])

        # Convert the TXName string to a sequence of token ids
        tx_name_ids = torch.tensor(self.tokenizer.encode(tx_names, padding='max_length', max_length=1024, add_special_tokens=True))
        # print(tx_name_ids.shape, padded_tx_names, len(padded_tx_names))
        # rssi_tensor = torch.tensor(padded_tx_rssis, dtype=torch.int64)

        return tx_name_ids, padded_tx_rssis, label
