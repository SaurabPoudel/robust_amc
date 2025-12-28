import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config.config import Config
from data.dataset import IQDataset
from data.generator import SignalGenerator
from models.moe_amc import MoEAMC, MoEAMCWithAnalysis
from models.expert_networks import ExpertCNN


def evaluate_model(model, test_loader, config):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_snrs = []
    
    with torch.no_grad():
        for signals, labels, snrs in tqdm(test_loader, desc="Evaluating"):
            signals = signals.to(config.DEVICE)
            outputs = model(signals)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snrs.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_snrs = np.array(all_snrs)
    
    # Overall accuracy
    accuracy = 100 * np.mean(all_predictions == all_labels)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    return all_predictions, all_labels, all_snrs, accuracy


def evaluate_by_snr(predictions, labels, snrs, config, snr_step=2):
    """Evaluate accuracy vs SNR"""
    snr_min, snr_max = config.SNR_RANGE
    snr_bins = np.arange(snr_min, snr_max + snr_step, snr_step)
    accuracies = []
    snr_centers = []
    
    for i in range(len(snr_bins) - 1):
        mask = (snrs >= snr_bins[i]) & (snrs < snr_bins[i + 1])
        if np.sum(mask) > 0:
            acc = 100 * np.mean(predictions[mask] == labels[mask])
            accuracies.append(acc)
            snr_centers.append((snr_bins[i] + snr_bins[i + 1]) / 2)
    
    return snr_centers, accuracies


def evaluate_by_modulation(predictions, labels, config):
    """Evaluate accuracy per modulation class"""
    accuracies = {}
    for i, mod in enumerate(config.MODULATIONS):
        mask = labels == i
        if np.sum(mask) > 0:
            acc = 100 * np.mean(predictions[mask] == labels[mask])
            accuracies[mod] = acc
    
    return accuracies


def plot_confusion_matrix(predictions, labels, config, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.MODULATIONS,
                yticklabels=config.MODULATIONS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_accuracy_vs_snr(snr_centers, accuracies, save_path=None):
    """Plot accuracy vs SNR curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(snr_centers, accuracies, marker='o', linewidth=2, markersize=8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Classification Accuracy vs SNR', fontsize=14)
    plt.ylim([0, 105])
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_modulation_accuracy(mod_accuracies, save_path=None):
    """Plot per-modulation accuracy"""
    mods = list(mod_accuracies.keys())
    accs = list(mod_accuracies.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(mods, accs, color='steelblue')
    plt.xlabel('Modulation', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy by Modulation Type', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim([0, 105])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def compare_with_baseline(moe_model, baseline_model, test_loader, config):
    """Compare MoE with baseline single model"""
    print("\nEvaluating MoE Model...")
    moe_pred, moe_labels, moe_snrs, moe_acc = evaluate_model(moe_model, test_loader, config)
    
    print("\nEvaluating Baseline Model...")
    baseline_pred, baseline_labels, baseline_snrs, baseline_acc = evaluate_model(
        baseline_model, test_loader, config
    )
    
    # Accuracy vs SNR comparison
    moe_snr_centers, moe_snr_accs = evaluate_by_snr(moe_pred, moe_labels, moe_snrs, config)
    baseline_snr_centers, baseline_snr_accs = evaluate_by_snr(
        baseline_pred, baseline_labels, baseline_snrs, config
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(moe_snr_centers, moe_snr_accs, marker='o', linewidth=2, 
             label='MoE AMC', markersize=8)
    plt.plot(baseline_snr_centers, baseline_snr_accs, marker='s', linewidth=2,
             label='Baseline CNN', markersize=8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('MoE vs Baseline: Accuracy vs SNR', fontsize=14)
    plt.legend(fontsize=11)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig('results/moe_vs_baseline.png')
    plt.show()
    
    print(f"\nMoE Overall Accuracy: {moe_acc:.2f}%")
    print(f"Baseline Overall Accuracy: {baseline_acc:.2f}%")
    print(f"Improvement: {moe_acc - baseline_acc:.2f}%")


def analyze_expert_contributions(model, test_loader, config):
    """Analyze how experts contribute to predictions"""
    model.eval()
    expert_usage = np.zeros(config.NUM_EXPERTS)
    expert_correct = np.zeros(config.NUM_EXPERTS)
    expert_total = np.zeros(config.NUM_EXPERTS)
    
    with torch.no_grad():
        for signals, labels, snrs in tqdm(test_loader, desc="Analyzing experts"):
            signals = signals.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            final_output, expert_outputs, gating_weights, snr_probs = model(
                signals, return_expert_outputs=True
            )
            
            # Track expert usage
            dominant_expert = torch.argmax(gating_weights, dim=1)
            for i in range(config.NUM_EXPERTS):
                mask = dominant_expert == i
                expert_usage[i] += mask.sum().item()
                
                if mask.sum() > 0:
                    expert_pred = torch.argmax(expert_outputs[mask, i, :], dim=1)
                    expert_correct[i] += (expert_pred == labels[mask]).sum().item()
                    expert_total[i] += mask.sum().item()
    
    # Print expert statistics
    print("\nExpert Analysis:")
    for i in range(config.NUM_EXPERTS):
        usage_pct = 100 * expert_usage[i] / expert_usage.sum()
        acc = 100 * expert_correct[i] / expert_total[i] if expert_total[i] > 0 else 0
        print(f"Expert {i} ({config.SNR_BINS[i]} SNR):")
        print(f"  Usage: {usage_pct:.2f}%")
        print(f"  Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    config = Config()
    
    # Generate test dataset
    print("Generating test dataset...")
    generator = SignalGenerator(
        samples_per_symbol=config.SAMPLES_PER_SYMBOL,
        num_symbols=config.NUM_SYMBOLS
    )
    
    test_signals, test_labels, test_snrs = generator.generate_dataset(
        modulations=config.MODULATIONS,
        snr_range=config.SNR_RANGE,
        samples_per_mod=500,
        channel='AWGN'
    )
    
    test_dataset = IQDataset(test_signals, test_labels, test_snrs)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Load MoE model
    print("\nLoading MoE model...")
    moe_model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=config.NUM_CLASSES,
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS,
        gating_mode='soft'
    ).to(config.DEVICE)
    
    moe_model.load_state_dict(torch.load(f"{config.MODEL_PATH}/moe_amc_best.pth"))
    
    # Evaluate
    predictions, labels, snrs, accuracy = evaluate_model(moe_model, test_loader, config)
    
    # Detailed analysis
    print("\n=== Accuracy by SNR ===")
    snr_centers, snr_accuracies = evaluate_by_snr(predictions, labels, snrs, config)
    for snr, acc in zip(snr_centers, snr_accuracies):
        print(f"SNR {snr:.1f} dB: {acc:.2f}%")
    
    print("\n=== Accuracy by Modulation ===")
    mod_accuracies = evaluate_by_modulation(predictions, labels, config)
    for mod, acc in mod_accuracies.items():
        print(f"{mod}: {acc:.2f}%")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(predictions, labels, config, 'results/confusion_matrix.png')
    plot_accuracy_vs_snr(snr_centers, snr_accuracies, 'results/accuracy_vs_snr.png')
    plot_modulation_accuracy(mod_accuracies, 'results/modulation_accuracy.png')
    
    # Expert analysis
    analyze_expert_contributions(moe_model, test_loader, config)
    
    print("\nEvaluation complete!")