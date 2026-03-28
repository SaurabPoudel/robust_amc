import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config.config import Config
from models.moe_amc import MoEAMC

class SDRDataset(Dataset):
    def __init__(self, filepath, window_size=128):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dat file not found at {filepath}")
            
        # Load the complex data
        print(f"Reading {filepath} with type complex64")
        data = np.fromfile(filepath, dtype=np.complex64)
        
        # Calculate exactly how many full windows we can get
        num_windows = len(data) // window_size
        
        # Slice and reshape
        self.signals = data[:num_windows * window_size].reshape((num_windows, window_size))
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        
        # Normalize to unit power: x / sqrt(E[|x|^2])
        power = np.mean(np.abs(signal)**2)
        if power > 0:
            signal = signal / np.sqrt(power)
            
        # Convert complex signal to 2-channel real representation (2, L)
        signal_tensor = np.stack([signal.real, signal.imag], axis=0)
        return torch.FloatTensor(signal_tensor)

def evaluate_sdr():
    config = Config()
    sdr_file = os.path.join(config.DATA_PATH, 'sdr_output.dat')
    
    print(f"Loading SDR data from {sdr_file}...")
    dataset = SDRDataset(sdr_file, window_size=config.SAMPLE_LENGTH)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Total windows (samples) to evaluate: {len(dataset)}")
    
    # Enable correct device processing
    mods = config.MODULATIONS
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=len(mods),
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS
    ).to(config.DEVICE)
    
    model_path = os.path.join(config.MODEL_PATH, 'moe_amc_best.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}. Exiting.")
        return
        
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for signals in tqdm(loader, desc="Evaluating SDR data"):
            signals = signals.to(config.DEVICE)
            outputs = model(signals)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            
    # Count predictions
    all_preds = np.array(all_preds)
    unique, counts = np.unique(all_preds, return_counts=True)
    
    print("\nPrediction Distribution:")
    for u, c in zip(unique, counts):
        print(f"{mods[u]}: {c} ({(c / len(all_preds)) * 100:.2f}%)")
        
    # Plot distribution
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([mods[u] for u in unique], counts, color='skyblue')
    plt.xlabel('Modulation Scheme')
    plt.ylabel('Count')
    plt.title('Modulation Predictions on SDR Data')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'sdr_predictions_distribution.png'))
    plt.close()
    
    print(f"\nSaved distribution plot to {os.path.join(config.RESULTS_PATH, 'sdr_predictions_distribution.png')}")

if __name__ == '__main__':
    evaluate_sdr()
