# Maritime Route Cost Forecaster

End-to-end ML pipeline for predicting container shipping freight rates (USD/FEU) across 4 major global trade routes.

## Overview

Integrates three data sources — **Xeneta shipping indices**, **NOAA weather data**, and **UNCTAD LSCI economic indices** — into a unified dataset with 50+ engineered features. Trains and evaluates **160 models** (Random Forest + XGBoost) with systematic hyperparameter sweeps across multiple train/test split ratios.

## Results

| Metric | Best Value | Model |
|--------|-----------|-------|
| **MAE** | **$217.40** | XGBoost (80/20) |
| **R²** | **0.964** | Random Forest (holdout) |
| **Dataset** | ~9,800 records | 4 routes × ~2,500 days |

## Project Structure

- `new dataset/` — Dataset assembly, cleaning, and feature engineering
- `ML Models/` — Model training (RF, XGBoost), evaluation, and visualization
- `website/` — FastAPI backend + HTML/CSS/JS frontend for live predictions
- `EDA/` — Exploratory data analysis and initial visualizations
- `Analysis/` — Correlation analysis and data design
- `internship/` — System analysis, activity, and data flow diagrams
- `oprational costs/` — UNCTAD LSCI economic indicator fetcher
- `mltest/` — Model comparison and stakeholder visuals

## Tech Stack

Python, XGBoost, scikit-learn, FastAPI, Pandas, NumPy

## Quick Start

```bash
# Install backend dependencies
cd website/backend
pip install -r requirements.txt

# Run the API server
python app.py
```
