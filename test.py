import torch
import numpy as np
from torch.utils.data import DataLoader

from config.config import Config
from data.dataset import IQDataset
from data.generator import SignalGenerator
from models.moe_amc import MoEAMC
from utils.visualization import plot_iq_samples, plot_signal_spectrum

def test_single_signal(model, signal, config):
    """Test model on a single signal"""
    model.eval()
    
    # Prepare signal
    signal_tensor = np.stack([signal.real, signal.imag], axis=0)
    signal_tensor = torch.FloatTensor(signal_tensor).unsqueeze(0)
    signal_tensor = signal_tensor.to(config.DEVICE)
    
    with torch.no_grad():
        output = model(signal_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][prediction].item()
    
    return prediction, confidence

def test_model_interactive():
    """Interactive testing of the model"""
    config = Config()
    
    # Load model
    print("Loading MoE model...")
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=config.NUM_CLASSES,
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS,
        gating_mode='soft'
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load(f"{config.MODEL_PATH}/moe_amc_best.pth"))
    print("Model loaded successfully!")
    
    # Initialize signal generator
    generator = SignalGenerator(
        samples_per_symbol=config.SAMPLES_PER_SYMBOL,
        num_symbols=config.NUM_SYMBOLS
    )
    
    print("\n=== Interactive Testing Mode ===")
    print("Available modulations:", config.MODULATIONS)
    
    while True:
        print("\n" + "="*50)
        mod_input = input("Enter modulation type (or 'quit' to exit): ").strip()
        
        if mod_input.lower() == 'quit':
            break
        
        if mod_input.upper() not in config.MODULATIONS:
            print(f"Invalid modulation! Choose from: {config.MODULATIONS}")
            continue
        
        snr_input = input("Enter SNR in dB (-10 to 20): ").strip()
        try:
            snr = float(snr_input)
            if snr < -10 or snr > 20:
                print("SNR out of range!")
                continue
        except ValueError:
            print("Invalid SNR value!")
            continue
        
        # Generate signal
        print(f"\nGenerating {mod_input.upper()} signal at {snr} dB SNR...")
        signal = generator.generate_signal(mod_input.upper())
        signal = generator.add_awgn(signal, snr)
        signal = signal / np.max(np.abs(signal))  # Normalize
        
        # Test
        prediction, confidence = test_single_signal(model, signal, config)
        predicted_mod = config.MODULATIONS[prediction]
        
        # Results
        print(f"\n{'='*50}")
        print(f"True Modulation:      {mod_input.upper()}")
        print(f"Predicted Modulation: {predicted_mod}")
        print(f"Confidence:           {confidence*100:.2f}%")
        print(f"Correct:              {'✓' if predicted_mod == mod_input.upper() else '✗'}")
        print(f"{'='*50}")
        
        # Ask if user wants to see visualization
        vis_input = input("\nShow signal visualization? (y/n): ").strip().lower()
        if vis_input == 'y':
            plot_iq_samples(
                np.array([signal]), 
                np.array([config.MODULATIONS.index(mod_input.upper())]),
                config.MODULATIONS,
                num_samples=1
            )

def batch_test():
    """Batch testing on generated signals"""
    config = Config()
    
    print("Generating test signals...")
    generator = SignalGenerator(
        samples_per_symbol=config.SAMPLES_PER_SYMBOL,
        num_symbols=config.NUM_SYMBOLS
    )
    
    # Generate test set
    test_signals, test_labels, test_snrs = generator.generate_dataset(
        modulations=config.MODULATIONS,
        snr_range=config.SNR_RANGE,
        samples_per_mod=100,
        channel='AWGN'
    )
    
    test_dataset = IQDataset(test_signals, test_labels, test_snrs)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    print("Loading model...")
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=config.NUM_CLASSES,
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS,
        gating_mode='soft'
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load(f"{config.MODEL_PATH}/moe_amc_best.pth"))
    
    # Test
    model.eval()
    correct = 0
    total = 0
    
    print("Testing...")
    with torch.no_grad():
        for signals, labels, snrs in test_loader:
            signals = signals.to(config.DEVICE)
            outputs = model(signals)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.cpu().eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f"\nBatch Test Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")

def test_expert_analysis():
    """Analyze expert behavior on test signals"""
    config = Config()
    
    print("Loading model...")
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=config.NUM_CLASSES,
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS,
        gating_mode='soft'
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load(f"{config.MODEL_PATH}/moe_amc_best.pth"))
    
    # Generate test signal
    generator = SignalGenerator()
    
    print("\n=== Expert Analysis ===")
    for snr in [-5, 5, 15]:
        print(f"\nTesting at SNR = {snr} dB")
        signal = generator.generate_signal('QPSK')
        signal = generator.add_awgn(signal, snr)
        signal = signal / np.max(np.abs(signal))
        
        signal_tensor = np.stack([signal.real, signal.imag], axis=0)
        signal_tensor = torch.FloatTensor(signal_tensor).unsqueeze(0).to(config.DEVICE)
        
        model.eval()
        with torch.no_grad():
            final_output, expert_outputs, gating_weights, snr_probs = model(
                signal_tensor, return_expert_outputs=True
            )
            
            print(f"  SNR probabilities: {snr_probs[0].cpu().numpy()}")
            print(f"  Gating weights: {gating_weights[0].cpu().numpy()}")
            print(f"  Final prediction: {config.MODULATIONS[torch.argmax(final_output[0]).item()]}")
            
            for i in range(config.NUM_EXPERTS):
                expert_pred = torch.argmax(expert_outputs[0, i, :]).item()
                print(f"  Expert {i} ({config.SNR_BINS[i]}): {config.MODULATIONS[expert_pred]}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'interactive':
            test_model_interactive()
        elif mode == 'batch':
            batch_test()
        elif mode == 'expert':
            test_expert_analysis()
        else:
            print("Usage: python test.py [interactive|batch|expert]")
    else:
        print("Select testing mode:")
        print("1. Interactive testing")
        print("2. Batch testing")
        print("3. Expert analysis")
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == '1':
            test_model_interactive()
        elif choice == '2':
            batch_test()
        elif choice == '3':
            test_expert_analysis()
        else:
            print("Invalid choice!")