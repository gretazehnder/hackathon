# Turbofan Engine Health Monitor
**Predictive maintenance for turbofan jet engines using LSTM (PyTorch) on NASA C-MAPSS dataset. Deployed on Google Cloud Platform.**

> University of Milan Hackathon · May 2026

---

## Live links

| Service | URL |
|---|---|
| **Streamlit Dashboard** | [engine-frontend-339853300574.europe-west1.run.app](https://engine-frontend-339853300574.europe-west1.run.app/) |
| **FastAPI Endpoint** | [engine-rul-api-339853300574.europe-west1.run.app/predict](https://engine-rul-api-339853300574.europe-west1.run.app/predict/) |
| **API Docs** | [/docs](https://engine-rul-api-339853300574.europe-west1.run.app/docs) |

---

## Project overview

Airlines perform maintenance on fixed schedules — regardless of actual engine condition. This leads to unnecessary costs or, worse, undetected failures.

This project predicts the **Remaining Useful Life (RUL)** of turbofan engines in real time, using sensor data from the last 50 flight cycles. When the predicted RUL drops below 90 cycles (~2 weeks of flights), the system triggers an alert.

---

## Model

- **Architecture:** LSTM, 2 layers, 128 hidden units (PyTorch)
- **Dataset:** NASA C-MAPSS FD001 — 100 turbofan engines, 17 sensors, 1 cycle = 1 flight
- **Target:** RUL — Remaining Useful Life (cycles to failure)
- **Test RMSE:** 12.81 cycles
- **Feature importance:** Gradient × Input

The LSTM reads the last 50 flight cycles of sensor history and predicts how many flights remain before failure — outperforming classical baselines (XGBoost RMSE: 17.6) by learning degradation patterns over time.

---

## Architecture

```
User → Streamlit Dashboard → FastAPI → Cloud Run → LSTM Model → Cloud SQL (PostgreSQL)
```

| Component | Technology | Description |
|---|---|---|
| Frontend | Streamlit | Interactive fleet monitoring dashboard |
| API | FastAPI | REST API endpoint for RUL predictions |
| Deploy | Cloud Run | Serverless container on GCP |
| Model | PyTorch LSTM | Pre-trained `.pt` file |
| Database | Cloud SQL PostgreSQL | Predictions history log |

---

## 📁 Repository Structure

```
hackathon/
├── api/                    # FastAPI backend
├── CMaps/                  # NASA C-MAPSS dataset (not tracked by git)
├── rul_lstm_pytorch.ipynb  # Full training notebook (EDA → LSTM → evaluation)
├── lstm_final.pt           # Pre-trained LSTM model weights
├── best_lstm.pt            # Best checkpoint during training
├── costestimatesummary.csv # Cost estimate summary
├── scaler_final.pkl        # MinMaxScaler fitted on training data
├── feature_list.pkl        # List of 17 live sensor features
├── hackathon pitch.pdf     # Project presentation slides
├── feature_importance.pkl  # Gradient × Input importance scores
├── engine_ranking.csv      # Lead time ranking for all 100 engines
├── pyproject.toml          # Poetry dependencies
├── poetry.lock             # Locked dependency versions
└── .env.edit               # Template for environment variables (copy to .env)
```

---

## Setup

### Prerequisites
- Python 3.11+
- [pyenv](https://github.com/pyenv/pyenv)
- [poetry](https://python-poetry.org)

### Install

```bash
git clone https://github.com/gretazehnder/hackathon
cd hackathon
poetry env use 3.11
poetry install --no-root
```

### Dataset

Download NASA C-MAPSS FD001 from Kaggle:
[kaggle.com/datasets/behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

Extract and place the `CMaps/` folder in the project root.

### Environment variables

```bash
cp .env.edit .env
# Fill in your Cloud SQL credentials in .env
```

---

## Monthly Cost Estimate (GCP)

| Service | Configuration | Cost |
|---|---|---|
| Cloud Run | 10,000 requests/month | $0.10 |
| Cloud SQL | PostgreSQL · db-f1-micro · 10GB | $9.37 |
| Cloud Storage | Model artifacts · 1GB | $0.00 |
| **Total** | | **$9.47 / month** |

*Estimate from Google Cloud Pricing Calculator · May 2026*

---

## Dataset

**NASA C-MAPSS FD001** — Commercial Modular Aero-Propulsion System Simulation

- 100 turbofan engines run to failure
- 21 sensors per engine (17 informative after removing constant sensors)
- 1 operating condition, 1 fault mode
- Target: RUL clipped at 125 cycles (standard C-MAPSS practice)

[Download from Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
