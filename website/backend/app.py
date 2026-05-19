"""
FastAPI Backend for Shipping Price Predictions
Serves predictions from last training date to today with streaming updates.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List
import uvicorn
from pathlib import Path

from predictor import get_predictor

app = FastAPI(
    title="Shipping Price Predictor API",
    description="Real-time shipping price predictions using XGBoost model",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files at /app path
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    @app.get("/app", include_in_schema=False)
    async def serve_frontend_index():
        """Serve frontend"""
        from fastapi.responses import FileResponse
        return FileResponse(str(frontend_dir / "index.html"))
    
    @app.get("/app/{full_path:path}", include_in_schema=False)
    async def serve_frontend_files(full_path: str):
        """Serve frontend static files"""
        from fastapi.responses import FileResponse
        file_path = frontend_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dir / "index.html"))

# Initialize predictor on startup
@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = get_predictor()
    print(f"Predictor initialized. Last training date: {predictor.last_training_date}")

# Request/Response models
class PredictionRequest(BaseModel):
    route: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class PredictionResponse(BaseModel):
    route: str
    date: str
    predicted_price: float
    confidence: float

# Routes
@app.get("/")
async def root():
    return {
        "message": "Shipping Price Predictor API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/api/model-info")
async def get_model_info():
    """Get model information and metrics"""
    return predictor.get_model_info()

@app.get("/api/historical-summary")
async def get_historical_summary():
    """Get historical price statistics"""
    return predictor.get_historical_summary()

@app.get("/api/predictions/today")
async def predict_today():
    """Get predictions for today"""
    predictions = predictor.predict_today()
    return {"predictions": predictions, "date": datetime.now().date().isoformat()}

@app.get("/api/predictions/range")
async def predict_range(
    start_date: str,
    end_date: str,
    route: Optional[str] = None
):
    """Get predictions for a date range"""
    try:
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        
        routes = [route] if route else None
        predictions = predictor.predict_range(start, end, routes)
        
        return {
            "predictions": predictions,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(predictions)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

@app.get("/api/predictions/stream")
async def stream_predictions():
    """
    Stream predictions from training cutoff to today.
    Shows the full gap: last training data → current date.
    """
    start_date = predictor.last_training_date.date()
    end_date = datetime.now().date()
    
    predictions = predictor.predict_range(start_date, end_date)
    
    # Group by route
    by_route = {}
    for pred in predictions:
        route = pred['route']
        if route not in by_route:
            by_route[route] = []
        by_route[route].append(pred)
    
    return {
        "training_cutoff": start_date.isoformat(),
        "today": end_date.isoformat(),
        "days_gap": (end_date - start_date).days,
        "predictions_by_route": by_route,
        "total_predictions": len(predictions)
    }

@app.get("/api/routes")
async def get_routes():
    """Get available shipping routes"""
    return {
        "routes": predictor.routes,
        "route_names": {
            "XSICFENE_FarEast_NorthEurope": "Far East → North Europe",
            "XSICNEFE_NorthEurope_FarEast": "North Europe → Far East",
            "XSICFEUW_FarEast_USWestCoast": "Far East → US West Coast",
            "XSICUWFE_USWestCoast_FarEast": "US West Coast → Far East"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "last_training_date": predictor.last_training_date.isoformat(),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
