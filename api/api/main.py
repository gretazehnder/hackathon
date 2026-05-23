import joblib
import torch
import torch.nn as nn
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

FEATURES = joblib.load('feature_list.pkl')
SEQ_LEN  = 50

scaler = joblib.load('scaler_final.pkl')
model  = LSTMRegressor(input_size=len(FEATURES))
model.load_state_dict(torch.load('lstm_final.pt', map_location='cpu'))
model.eval()

app = FastAPI(
    title="Engine RUL Prediction API",
    version="0.1.0",
    description="Predicts Remaining Useful Life (RUL) of airplane engines."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class CycleReading(BaseModel):
    T24: float; T30: float; T50: float; P15: float; P30: float
    Nf: float;  Nc: float;  Ps30: float; phi: float; NRf: float
    NRc: float; BPR: float; htBleed: float; W31: float; W32: float
    setting_1: float; setting_2: float

class EngineSequence(BaseModel):
    cycles: List[CycleReading]

@app.get("/")
def read_root():
    return {"message": "Engine RUL API — POST /predict/ with 50 cycles of sensor data."}

@app.post("/predict/")
def predict(sequence: EngineSequence) -> dict:
    if len(sequence.cycles) != SEQ_LEN:
        raise HTTPException(422, detail=f"Expected {SEQ_LEN} cycles, got {len(sequence.cycles)}")
    X = np.array([[getattr(c, f) for f in FEATURES] for c in sequence.cycles], dtype=np.float32)
    X_scaled = scaler.transform(X)
    X_tensor  = torch.tensor(X_scaled).unsqueeze(0)
    with torch.no_grad():
        rul = model(X_tensor).item()
    rul = round(max(0.0, rul), 2)
    return {"rul": rul, "warning": rul < 90}
