import os
import torch

class Config:
    # Dataset parameters
    MODULATIONS = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'GFSK', 'CPFSK', 'PAM4']
    NUM_CLASSES = len(MODULATIONS)
    SAMPLES_PER_SYMBOL = 8
    NUM_SYMBOLS = 128
    SAMPLE_LENGTH = SAMPLES_PER_SYMBOL * NUM_SYMBOLS  # 1024
    
    # SNR parameters
    SNR_RANGE = (-10, 20)  # dB
    SNR_LOW = (-10, 0)
    SNR_MID = (0, 10)
    SNR_HIGH = (10, 20)
    SNR_BINS = ['low', 'mid', 'high']
    NUM_EXPERTS = len(SNR_BINS)
    
    # Channel models
    CHANNELS = ['AWGN', 'Rayleigh', 'Rician']
    
    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5
    
    # Model parameters
    SNR_ESTIMATOR_HIDDEN = [128, 64]
    EXPERT_CNN_FILTERS = [64, 128, 256]
    GATING_HIDDEN = [128, 64]
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'generated')
    MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints')
    LOG_PATH = os.path.join(BASE_DIR, 'logs')
    
    @classmethod
    def get_snr_bin(cls, snr):
        """Convert SNR value to bin index"""
        if snr < 0:
            return 0  # low
        elif snr < 10:
            return 1  # mid
        else:
            return 2  # high