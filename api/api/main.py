import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import psycopg2
from datetime import datetime
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


# ── Load model and scaler ──
FEATURES = joblib.load('feature_list.pkl')
SEQ_LEN  = 50
scaler   = joblib.load('scaler_final.pkl')
model    = LSTMRegressor(input_size=len(FEATURES))
model.load_state_dict(torch.load('lstm_final.pt', map_location='cpu'))
model.eval()


# ── Database connection ──
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS")
    )


def init_db():
    """Create predictions table if it doesn't exist."""
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          SERIAL PRIMARY KEY,
                engine_id   INTEGER,
                rul         FLOAT,
                warning     BOOLEAN,
                timestamp   TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB init error: {e}")


def save_prediction(engine_id: int, rul: float, warning: bool):
    """Save a prediction to the database."""
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (engine_id, rul, warning, timestamp) VALUES (%s, %s, %s, %s)",
            (engine_id, rul, warning, datetime.now())
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB save error: {e}")


# ── FastAPI app ──
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


@app.on_event("startup")
def startup_event():
    init_db()


# ── Schemas ──
class CycleReading(BaseModel):
    T24: float; T30: float; T50: float; P15: float; P30: float
    Nf: float;  Nc: float;  Ps30: float; phi: float; NRf: float
    NRc: float; BPR: float; htBleed: float; W31: float; W32: float
    setting_1: float; setting_2: float


class EngineSequence(BaseModel):
    engine_id: int = 0
    cycles: List[CycleReading]


# ── Endpoints ──
@app.get("/")
def read_root():
    return {"message": "Engine RUL API — POST /predict/ with 50 cycles of sensor data."}


@app.post("/predict/")
def predict(sequence: EngineSequence) -> dict:
    if len(sequence.cycles) != SEQ_LEN:
        raise HTTPException(422, detail=f"Expected {SEQ_LEN} cycles, got {len(sequence.cycles)}")

    X        = np.array([[getattr(c, f) for f in FEATURES] for c in sequence.cycles], dtype=np.float32)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled).unsqueeze(0)

    with torch.no_grad():
        rul = model(X_tensor).item()

    rul     = round(max(0.0, rul), 2)
    warning = rul < 90

    # Save to database
    save_prediction(sequence.engine_id, rul, warning)

    return {"rul": rul, "warning": warning}


@app.get("/predictions/")
def get_predictions():
    """Return last 100 predictions from the database."""
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            "SELECT engine_id, rul, warning, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 100"
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {"engine_id": r[0], "rul": r[1], "warning": r[2], "timestamp": str(r[3])}
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(500, detail=str(e))