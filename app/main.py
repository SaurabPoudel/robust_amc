import os
import numpy as np
import torch
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
from pathlib import Path

# Add parent directory to path to import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.moe_amc import MoEAMC
from config.config import Config
from data.generator import SignalGenerator

app = FastAPI(title="Robust AMC API")

# Ensure directories exist
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Create a placeholder file in static directory
with open(os.path.join(STATIC_DIR, '.gitkeep'), 'w') as f:
    pass

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load configuration and model
config = Config()
model = None
generator = None

class PredictionRequest(BaseModel):
    iq_data: list[list[float]]
    snr: float = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    expert_used: str
    snr_estimate: float

@app.on_event("startup")
async def load_model():
    """Load the trained model and signal generator"""
    global model, generator
    
    try:
        # Initialize model
        model = MoEAMC(
            num_experts=config.NUM_EXPERTS,
            num_classes=config.NUM_CLASSES,
            input_channels=2,
            expert_filters=config.EXPERT_CNN_FILTERS,
            gating_mode='soft'
        ).to(config.DEVICE)
        
        # Get absolute path to model checkpoint
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
        model_path = os.path.join(model_dir, 'moe_amc_best.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "Please make sure to train the model first by running 'python train.py'"
            )
            
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        
        # Initialize signal generator
        generator = SignalGenerator(
            samples_per_symbol=config.SAMPLES_PER_SYMBOL,
            num_symbols=config.NUM_SYMBOLS
        )
        print("Model and generator loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTo train the model, run:")
        print("1. python train.py")
        print("2. Then start the API with: python -m uvicorn app.main:app --reload")
        raise e

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request, "modulations": config.MODULATIONS})

@app.post("/predict", response_model=PredictionResponse)
async def predict_modulation(request: PredictionRequest):
    """API endpoint for modulation prediction"""
    # Convert input to numpy array - input is [[r, i], [r, i], ...]
    if not request.iq_data:
        raise HTTPException(status_code=400, detail="Empty IQ data")
        
    iq_data = np.array(request.iq_data, dtype=np.float32)
    
    # Ensure we have the correct length
    target_length = config.SAMPLE_LENGTH  # 1024
    current_length = len(iq_data)
    
    if current_length < target_length:
        # Pad with zeros
        padding = np.zeros((target_length - current_length, 2), dtype=np.float32)
        iq_data = np.vstack([iq_data, padding])
    elif current_length > target_length:
        # Truncate
        iq_data = iq_data[:target_length]
    
    # iq_data shape is now (1024, 2)
    
    # Convert to tensor: shape (1, 2, 1024)
    # iq_data[:, 0] is real part, iq_data[:, 1] is imag part
    iq_tensor = torch.tensor(np.stack([iq_data[:, 0], iq_data[:, 1]], axis=0), 
                            dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    
    # Make prediction
    with torch.no_grad():
        outputs, expert_outputs, gating_weights, snr_probs = model(iq_tensor, return_expert_outputs=True)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
        
        # Get expert with highest weight (gating_weights is already squeezed in return)
        expert_idx = torch.argmax(gating_weights[0]).item()
        expert_used = config.SNR_BINS[expert_idx]
        
        # Estimate SNR from the snr_probs (the expert probabilities indicate SNR bin)
        # Low SNR: -10 to 0, Mid SNR: 0 to 10, High SNR: 10 to 20
        snr_ranges = [(-10, 0), (0, 10), (10, 20)]
        estimated_snr = 0.0
        for i, (snr_low, snr_high) in enumerate(snr_ranges):
            prob = snr_probs[0, i].item()
            estimated_snr += prob * (snr_low + snr_high) / 2
        
        return {
            "prediction": config.MODULATIONS[pred.item()],
            "confidence": confidence.item(),
            "expert_used": expert_used,
            "snr_estimate": estimated_snr
        }

@app.post("/generate")
async def generate_signal(request: Request):
    """Generate a signal with the specified modulation and SNR"""
    try:
        # Parse form data
        form_data = await request.form()
        modulation = form_data.get("modulation")
        snr_str = form_data.get("snr", "0.0")
        try:
            snr = float(snr_str)
        except ValueError:
            snr = 0.0
            
        print(f"Received request - Modulation: {modulation}, SNR: {snr}")
        
        if modulation not in config.MODULATIONS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid modulation. Must be one of: {config.MODULATIONS}"}
            )
        
        # Generate signal
        signal = generator.generate_signal(modulation)
        
        # Add noise if SNR is specified
        if snr is not None:
            signal = generator.add_awgn(signal, snr)
        
        # Normalize
        signal = signal / np.max(np.abs(signal) + 1e-10)  # Add small value to avoid division by zero
        
        # Convert complex to list of [real, imag] pairs
        iq_data = [[s.real, s.imag] for s in signal]
        
        return {
            "iq_data": iq_data,
            "modulation": modulation,
            "snr": snr
        }
        
    except Exception as e:
        print(f"Error in generate_signal: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error generating signal: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
