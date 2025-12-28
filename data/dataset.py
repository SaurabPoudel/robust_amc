import torch
from torch.utils.data import Dataset
import numpy as np

class IQDataset(Dataset):
    def __init__(self, signals, labels, snr_values):
        """
        Args:
            signals: Complex I/Q samples (N, L) where L is sample length
            labels: Modulation class labels (N,)
            snr_values: SNR values in dB (N,)
        """
        self.signals = signals
        self.labels = labels
        self.snr_values = snr_values
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        
        # Convert complex signal to 2-channel real representation
        signal_tensor = np.stack([signal.real, signal.imag], axis=0)
        signal_tensor = torch.FloatTensor(signal_tensor)
        
        label = torch.LongTensor([self.labels[idx]])[0]
        snr = torch.FloatTensor([self.snr_values[idx]])[0]
        
        return signal_tensor, label, snr


class SNRStratifiedDataset(Dataset):
    def __init__(self, signals, labels, snr_values, snr_range):
        """
        Dataset filtered by SNR range
        
        Args:
            signals: Complex I/Q samples
            labels: Modulation class labels
            snr_values: SNR values in dB
            snr_range: Tuple (min_snr, max_snr) to filter
        """
        mask = (snr_values >= snr_range[0]) & (snr_values < snr_range[1])
        self.signals = signals[mask]
        self.labels = labels[mask]
        self.snr_values = snr_values[mask]
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        signal_tensor = np.stack([signal.real, signal.imag], axis=0)
        signal_tensor = torch.FloatTensor(signal_tensor)
        
        label = torch.LongTensor([self.labels[idx]])[0]
        snr = torch.FloatTensor([self.snr_values[idx]])[0]
        
        return signal_tensor, label, snr