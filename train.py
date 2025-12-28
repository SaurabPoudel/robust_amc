import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import json

from config.config import Config
from data.dataset import IQDataset, SNRStratifiedDataset
from data.generator import SignalGenerator
from models.moe_amc import MoEAMC
from models.expert_networks import ExpertCNN
from models.snr_estimator import SNREstimator
from utils.metrics import calculate_accuracy, calculate_confusion_matrix
from utils.visualization import plot_training_curves, plot_confusion_matrix

def train_snr_estimator(train_loader, val_loader, config, num_epochs=50):
    """Train SNR estimator separately"""
    print("Training SNR Estimator...")
    
    model = SNREstimator(
        input_channels=2,
        hidden_dims=config.SNR_ESTIMATOR_HIDDEN,
        output_dim=config.NUM_EXPERTS
    ).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                          weight_decay=config.WEIGHT_DECAY)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for signals, labels, snrs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            signals = signals.to(config.DEVICE)
            snrs = snrs.to(config.DEVICE)
            
            # Convert SNR to bin labels
            snr_bins = torch.tensor([config.get_snr_bin(s.item()) for s in snrs]).to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, snr_bins)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += snr_bins.size(0)
            correct += predicted.eq(snr_bins).sum().item()
        
        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels, snrs in val_loader:
                signals = signals.to(config.DEVICE)
                snrs = snrs.to(config.DEVICE)
                snr_bins = torch.tensor([config.get_snr_bin(s.item()) for s in snrs]).to(config.DEVICE)
                
                outputs = model(signals)
                loss = criterion(outputs, snr_bins)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += snr_bins.size(0)
                correct += predicted.eq(snr_bins).sum().item()
        
        val_acc = 100. * correct / total
        val_losses.append(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(config.MODEL_PATH, exist_ok=True)
            torch.save(model.state_dict(), f"{config.MODEL_PATH}/snr_estimator_best.pth")
    
    return model, train_losses, val_losses


def train_expert(expert_id, train_loader, val_loader, config, num_epochs=100):
    """Train individual expert for specific SNR range"""
    print(f"Training Expert {expert_id} ({config.SNR_BINS[expert_id]} SNR)...")
    
    model = ExpertCNN(
        input_channels=2,
        num_classes=config.NUM_CLASSES,
        filters=config.EXPERT_CNN_FILTERS
    ).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                          weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=10)
    
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for signals, labels, snrs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            signals = signals.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels, snrs in val_loader:
                signals = signals.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                outputs = model(signals)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        val_accs.append(val_acc)
        
        scheduler.step(val_acc)
        
        print(f"Expert {expert_id} Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(config.MODEL_PATH, exist_ok=True)
            torch.save(model.state_dict(), 
                      f"{config.MODEL_PATH}/expert_{expert_id}_best.pth")
    
    return model, train_losses, val_accs


def train_moe_system(train_loader, val_loader, config, num_epochs=100):
    """Train complete MoE system end-to-end"""
    print("Training Complete MoE AMC System...")
    
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=config.NUM_CLASSES,
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS,
        gating_mode='soft'
    ).to(config.DEVICE)
    
    # Load pre-trained components if available
    if os.path.exists(f"{config.MODEL_PATH}/snr_estimator_best.pth"):
        model.snr_estimator.load_state_dict(
            torch.load(f"{config.MODEL_PATH}/snr_estimator_best.pth")
        )
        print("Loaded pre-trained SNR estimator")
    
    for i in range(config.NUM_EXPERTS):
        expert_path = f"{config.MODEL_PATH}/expert_{i}_best.pth"
        if os.path.exists(expert_path):
            model.experts[i].load_state_dict(torch.load(expert_path))
            print(f"Loaded pre-trained expert {i}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE/2,
                          weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=10)
    
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for signals, labels, snrs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            signals = signals.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels, snrs in val_loader:
                signals = signals.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                outputs = model(signals)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        val_accs.append(val_acc)
        
        scheduler.step(val_acc)
        
        print(f"MoE Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{config.MODEL_PATH}/moe_amc_best.pth")
            print(f"Saved best model with val acc: {val_acc:.2f}%")
    
    return model, train_losses, val_accs


if __name__ == "__main__":
    config = Config()
    
    # Generate or load dataset
    print("Generating dataset...")
    generator = SignalGenerator(
        samples_per_symbol=config.SAMPLES_PER_SYMBOL,
        num_symbols=config.NUM_SYMBOLS
    )
    
    signals, labels, snrs = generator.generate_dataset(
        modulations=config.MODULATIONS,
        snr_range=config.SNR_RANGE,
        samples_per_mod=1000,
        channel='AWGN'
    )
    
    # Split dataset
    n = len(signals)
    indices = np.random.permutation(n)
    train_end = int(config.TRAIN_SPLIT * n)
    val_end = train_end + int(config.VAL_SPLIT * n)
    
    train_dataset = IQDataset(signals[indices[:train_end]], 
                              labels[indices[:train_end]],
                              snrs[indices[:train_end]])
    val_dataset = IQDataset(signals[indices[train_end:val_end]],
                           labels[indices[train_end:val_end]],
                           snrs[indices[train_end:val_end]])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Train pipeline
    # 1. Train SNR estimator
    snr_model, snr_train_loss, snr_val_loss = train_snr_estimator(
        train_loader, val_loader, config, num_epochs=50
    )
    
    # 2. Train experts for each SNR range
    snr_ranges = [config.SNR_LOW, config.SNR_MID, config.SNR_HIGH]
    for expert_id, snr_range in enumerate(snr_ranges):
        expert_train = SNRStratifiedDataset(
            signals[indices[:train_end]],
            labels[indices[:train_end]],
            snrs[indices[:train_end]],
            snr_range
        )
        expert_val = SNRStratifiedDataset(
            signals[indices[train_end:val_end]],
            labels[indices[train_end:val_end]],
            snrs[indices[train_end:val_end]],
            snr_range
        )
        
        expert_train_loader = DataLoader(expert_train, batch_size=config.BATCH_SIZE, shuffle=True)
        expert_val_loader = DataLoader(expert_val, batch_size=config.BATCH_SIZE, shuffle=False)
        
        train_expert(expert_id, expert_train_loader, expert_val_loader, config)
    
    # 3. Train complete MoE system
    moe_model, moe_train_loss, moe_val_acc = train_moe_system(
        train_loader, val_loader, config, num_epochs=50
    )
    
    print("Training complete!")